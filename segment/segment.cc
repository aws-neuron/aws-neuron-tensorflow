/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/python/neuron/segment/segment.h"
#include "tensorflow/python/neuron/segment/union_find.h"

namespace tensorflow {
namespace kaena {
namespace segment {

namespace {

bool CanContractEdge(const tensorflow::Edge* edge,
                     const tensorflow::Graph& graph) {
  const tensorflow::Node* src = edge->src();
  const tensorflow::Node* dst = edge->dst();

  // Can't contract edge if doing so would cause a cycle in the
  // graph. So, if there is a directed path from 'src' to 'dst', other
  // than 'edge' (or any other direct edge from 'src' to 'dst'), then
  // combining 'src' and 'dst' will cause a cycle along that path.
  //
  // In practice, to avoid modifying the graph and to take advantage
  // of existing graph functions, we perform an equivalent.
  //   1. Get all nodes incoming to 'dst', excluding 'src'
  //   2. Reverse DFS from those nodes
  //   3. If reverse DFS reaches 'src' then we have a cycle
  std::vector<tensorflow::Node*> dfs_start_nodes;
  for (tensorflow::Node* node : dst->in_nodes()) {
    if (node != src) {
      dfs_start_nodes.push_back(node);
    }
  }

  bool is_cycle = false;
  if (!dfs_start_nodes.empty()) {
    tensorflow::ReverseDFSFrom(graph, dfs_start_nodes, {},
                               [&is_cycle, src](tensorflow::Node* node) {
                                 if (node == src) {
                                   is_cycle = true;
                                 }
                               });
  }

  return !is_cycle;
}

void ContractEdge(tensorflow::Edge* edge, tensorflow::Graph* graph,
                  std::vector<const tensorflow::Edge*>* remove_edges) {
  // Transfer all inputs and outputs of 'dst' to 'src' except edges
  // connecting the two.
  tensorflow::Node* src = edge->src();
  tensorflow::Node* dst = edge->dst();

  // We can use '0' for input/output index because we don't need them
  // to be accurate for the way we are using the graph.
  std::vector<const tensorflow::Edge*> in_edges(dst->in_edges().begin(),
                                                dst->in_edges().end());
  for (const tensorflow::Edge* in_edge : in_edges) {
    if (in_edge->IsControlEdge()) {
      if (in_edge->src() != src) {
        tensorflow::Edge* e = const_cast<tensorflow::Edge*>(in_edge);
        graph->AddControlEdge(e->src(), src);
      }
    } else {
      if (in_edge->src() != src) {
        tensorflow::Edge* e = const_cast<tensorflow::Edge*>(in_edge);
        if (e->src() == graph->source_node()) {
          graph->AddEdge(e->src(), e->src_output(), src,
                         tensorflow::Graph::kControlSlot);
        } else {
          graph->AddEdge(e->src(), e->src_output(), src, 0 /* input index */);
        }
      }
    }
  }

  std::vector<const tensorflow::Edge*> out_edges(dst->out_edges().begin(),
                                                 dst->out_edges().end());
  for (const tensorflow::Edge* out_edge : out_edges) {
    if (out_edge->IsControlEdge()) {
      tensorflow::Edge* e = const_cast<tensorflow::Edge*>(out_edge);
      graph->AddControlEdge(src, e->dst());
    } else {
      tensorflow::Edge* e = const_cast<tensorflow::Edge*>(out_edge);
      if (e->dst() == graph->sink_node()) {
        VLOG(1) << " edge to sink node " << src->name() << " -> "
                << e->dst()->name();
        graph->AddEdge(src, tensorflow::Graph::kControlSlot, e->dst(),
                       e->dst_input());
      } else {
        graph->AddEdge(src, 0 /* output index */, e->dst(), e->dst_input());
      }
    }
  }

  // Return the edges that must be removed to disconnect 'dst' from
  // the graph. We don't actually remove 'dst' since the caller holds
  // references to all the nodes.
  for (const auto& in_edge : dst->in_edges()) {
    remove_edges->push_back(in_edge);
  }
  for (const auto& out_edge : dst->out_edges()) {
    remove_edges->push_back(out_edge);
  }
}

bool isPrefix(std::string shorter, std::string longer) {
  VLOG(2) << " Checking if " << shorter << " is a prefix of " << longer;
  if (shorter.size() >= longer.size()) {
    return false;
  }
  auto res = std::mismatch(shorter.begin(), shorter.end(), longer.begin());
  return (res.first == shorter.end());
}

struct Label_Node {
  std::string name;  // frame name of each node
  bool valid;        // if label has all whitelisted ids.
  std::set<std::string>
      labels;  // list of valid labels including itself and children
  std::vector<Label_Node*> children;
};

/*
 A label tree is made from label names. Since only labels crossing
 "Enter" are interesting for us. Making Tree from "Enter" frame names.
 For example:
 label_names = [_main_/Enter_f0, _main_/Enter_f0/Enter_f1, _main_/Enter_f2]
 The tree will look like:
                      _main_
                      /    \
        _main_/Enter_f0    _main_/Enter_f2
            /
 _main_/Enter_f0/Enter_f1

*/
Label_Node* ConstructLabelTree(std::set<std::string> ignore_labels,
                               std::vector<std::string> enter_frame_names) {
  VLOG(2) << " Constructing label tree ...........";
  Label_Node* root_node = new Label_Node();
  root_node->valid = false;
  root_node->name = "_main_";

  VLOG(2) << "Frame names: ";
  for (auto name : enter_frame_names) {
    VLOG(2) << name;
  }

  // Sort the frame names so that parent and child are always in order.
  // i.e parent followed by child.
  std::sort(enter_frame_names.begin(), enter_frame_names.end());

  // Iterate through all the frame names.
  // For each frame name find the parent in the tree and make
  // this label its child.
  for (auto node_names : enter_frame_names) {
    std::size_t found = node_names.find_last_of("/");
    std::string parent_name = node_names.substr(0, found);

    Label_Node* new_node = new Label_Node();
    new_node->valid = (ignore_labels.find(node_names) == ignore_labels.end());
    new_node->name = node_names;

    Label_Node* curr_root = root_node;
    while (1) {
      VLOG(2) << " node_name: " << node_names;
      VLOG(2) << " parent_name:" << parent_name;
      if (curr_root->name == parent_name) {
        VLOG(2) << "Pushed: " << new_node->name << " to: " << curr_root->name;
        curr_root->children.push_back(new_node);
        break;
      }
      for (auto child : curr_root->children) {
        if (isPrefix(child->name, node_names)) {
          curr_root = child;
          break;
        }
      }  // for
    }    // while
  }      // for names

  return root_node;
}

/*
  Since for a nested loop, all labels will be seperate  i.e outer loop  and all
  nested loops will have seperate label , we have to find the list of loop
  labels that has everything inside it as valid (whitelisted).

  For example:
    A/B is valid
    A/B/C is invalid
    A/B/S is valid

    The valid labels is [A/B/S].

  Another case:
    A/B is valid
    A/B/C is valid
    A/B/S is valid

    The valid labels is [A/B, A/B/C, A/B/S]

  To find valid segment labels.
  Iterate through the nodes of tree from leaf nodes.
  At each node:
  1. If the (node is invalid) or (node is valid but one or more of its children
  is invalid). Push all the labels on the node to the final result.
  2. else: add the curr node label and all the children label to the node label
  list, as shown in the second example above.
  3. For the leaf node, if itself is valid, just add it to its "label" vector.

  At the end of recursion, we will have all the valid nodes added to the result.
*/
void FindValidSegments(Label_Node* curr_node,
                       std::vector<std::set<std::string>>& valid_segments) {
  if (curr_node->children.size() == 0) {
    if (curr_node->valid) {
      curr_node->labels.insert(curr_node->name);
    }
    return;
  }

  bool jointValid = true;
  for (auto node : curr_node->children) {
    FindValidSegments(node, valid_segments);
    jointValid &= node->valid;
  }

  if (!curr_node->valid || (curr_node->valid && !jointValid)) {
    for (auto node : curr_node->children) {
      if (node->labels.size()) {
        valid_segments.push_back(node->labels);
      }
    }
    curr_node->valid = false;
  } else {
    curr_node->labels.insert(curr_node->name);
    for (auto node : curr_node->children) {
      curr_node->labels.insert(node->name);
    }
    curr_node->valid = true;
  }
}

void printTree(Label_Node* root_node) {
  VLOG(2) << root_node->name;
  VLOG(2) << root_node->valid;
  for (auto child : root_node->children) {
    printTree(child);
  }
}

void DeleteLabelTree(Label_Node* curr_node) {
  if (curr_node != NULL && curr_node->children.size() == 0) {
    delete curr_node;
    return;
  }

  for (auto node : curr_node->children) {
    DeleteLabelTree(node);
  }

  if (curr_node != NULL) {
    delete curr_node;
  }
}

void FindValidLoopSegments(std::set<std::string>& ignore_labels,
                           std::vector<std::string>& enter_frame_names,
                           std::vector<std::set<std::string>>& valid_segments) {
  // Construct Tree from label names.
  Label_Node* root_node = ConstructLabelTree(ignore_labels, enter_frame_names);
  VLOG(2) << " Tree .....";
  printTree(root_node);

  FindValidSegments(root_node, valid_segments);
  DeleteLabelTree(root_node);
}

/*
 To find loop segments, we iterate through the graph in DFS and provide a label
 to each node. A label for a node is same as parent label till we hit an "Enter"
 Node. At every "Enter", we add the frame name of the "Enter" op to the parent
 label After every "Exit" we remove the last label added to the name. The "Exit"
 will still have the same name as its parent. For example.
      Add(a)->Sum(b)->Enter(c)->.........->Exit(x)->Add(y)
      The itme in the () is name of the node.
      The starting label is always "_main_".
      Maitaining a map of label names -> to all nodes with same label

      _main_ -> a, b, y
      _main_/fc -> c,.....,x   where 'fc' is frame name of Enter provided by TF.

For a nested loop, the outer loop and inner loop will have  seperate entries.
The outer label will have only nodes that are inside that loop but outside the
nested loops.Each nested loop will have respective entries in the label map.
There will be no duplicate entries across the differet labels including the
nested labels for nested loops.
 */
static void GetLoopSegments(
    tensorflow::Graph& graph, std::set<std::string>* op_whitelist,
    std::set<std::string>* op_cpu, std::set<std::string>* op_inferentia,
    SegmentNodesVector* loop_segments, std::set<string>& exclude_list,
    std::set<string>& loop_breaking_ops,
    const std::function<bool(const tensorflow::Node*,
                             const std::set<std::string>&,
                             const std::set<std::string>&,
                             const std::set<std::string>&)>& candidate_fn) {
  // maps a label to all the node_ids with having the same label.
  std::unordered_map<std::string, std::vector<int>> label_map;

  // Label generation start .......
  std::vector<std::pair<Node*, string> > node_stack;
  std::vector<bool> visited(graph.num_node_ids(), false);

  tensorflow::Node* src_node = graph.source_node();
  VLOG(2) << "Src node:" << src_node->name();
  node_stack.push_back(std::make_pair(src_node, "_main_"));
  while (!node_stack.empty()) {
    tensorflow::Node* curr_node = node_stack.back().first;
    std::string parent_label = node_stack.back().second;
    node_stack.pop_back();
    visited[curr_node->id()] = true;

    if (curr_node->op_def().name() == "Enter") {
      std::string frame_name;
      GetNodeAttr(curr_node->def(), "frame_name", &frame_name);
      // changing all "/" in frame_name to "-"
      std::replace(frame_name.begin(), frame_name.end(), '/', '-');

      // Adding frame_name to parent label
      parent_label += "/" + frame_name;
      label_map[parent_label].push_back(curr_node->id());
      VLOG(2) << " frame: " << parent_label
              << " node: " << graph.FindNodeId(curr_node->id())->name();
    } else if (curr_node->op_def().name() == "Exit") {
      // Exit will still have the parent frame info.
      label_map[parent_label].push_back(curr_node->id());
      VLOG(2) << " frame: " << parent_label
              << " node: " << graph.FindNodeId(curr_node->id())->name();
      // On Exit, the remove the last frame info
      std::size_t found = parent_label.find_last_of("/");
      parent_label = parent_label.substr(0, found);
    } else {
      // No point maintaining the nodes with label "_main_", as it means, its
      // not part of loop
      if (parent_label != "_main_") {
        label_map[parent_label].push_back(curr_node->id());
        VLOG(2) << " frame: " << parent_label
                << " node: " << graph.FindNodeId(curr_node->id())->name();
      }
    }

    for (auto edge : curr_node->out_edges()) {
      if (!visited[edge->dst()->id()]) {
        node_stack.push_back(std::make_pair(edge->dst(), parent_label));
      }
    }
  }  // stack

  // ignore_labels are list of labels that has atleast one node_id that is not
  // in whitelist
  std::set<std::string> ignore_labels;
  std::vector<std::string> enter_frame_names;
  for (auto kv : label_map) {
    enter_frame_names.push_back(kv.first);
    for (auto idx : kv.second) {
      tensorflow::Node* node = graph.FindNodeId(idx);
      if (!candidate_fn(node, *op_whitelist, *op_cpu, *op_inferentia)) {
        VLOG(1) << " Not in whitelist: " << node->name()
                << " op: " << node->op_def().name();
        loop_breaking_ops.insert(node->op_def().name());
        ignore_labels.insert(kv.first);
        break;
      }
    }
  }

  // Label generation finished ..............

  VLOG(2) << " Number of possible loop_segments: " << label_map.size();

  // List of segments that contain only node_id's that are in whitelisted
  std::vector<std::set<std::string>> valid_segments;

  FindValidLoopSegments(ignore_labels, enter_frame_names, valid_segments);

  // valid_segment is only a list of valid labels
  // extracting all the nodes under these labels using label map.
  std::set<std::string> name_segment;
  for (auto label_segment : valid_segments) {
    VLOG(2) << "Segment: ...........";
    for (auto label_names : label_segment) {
      for (auto node_id : label_map[label_names]) {
        std::string node_name = graph.FindNodeId(node_id)->name();
        VLOG(2) << node_name;
        name_segment.insert(node_name);
      }
    }
    loop_segments->emplace_back(name_segment);
    name_segment.clear();
  }

  // This is needed for estimator for handling control edges
  // also can be used as optimization to keep complete loop local
  // if cannot be send.
  // put nodes associated with ignore_labels to exclude_list
  // name_segment.clear();
  // for (auto label_names : ignore_labels) {
  //   for (auto node_id : label_map[label_names]) {
  //     std::string node_name = graph.FindNodeId(node_id)->name();
  //     name_segment.insert(node_name);
  //   }
  // }
  // exclude_list.insert(name_segment.begin(), name_segment.end());

  VLOG(2) << " Number of valid loop_segments: " << valid_segments.size();
  for (auto frame_name : enter_frame_names) {
    VLOG(2) << frame_name;
  }

  // if any of the loop segment has non whitelisted ops, then we want keep those
  // loops locally and send just the body to server if possible. So removing all
  // the loop ops from whitelist.
  if (ignore_labels.size()) {
    VLOG(2) << " Removing Cond ops from whitelist";
    op_whitelist->erase("Enter");
    op_whitelist->erase("Merge");
    op_whitelist->erase("LoopCond");
    op_whitelist->erase("Exit");
    op_whitelist->erase("Switch");
    op_whitelist->erase("NextIteration");
  }
}

}  // namespace

// Remove NextIteration outedge to get rid of cycles
void RemoveCycleEdge(tensorflow::Graph& graph) {
  std::vector<std::pair<int, int>> nextiteration_edges;
  for (int i = 0; i < graph.num_node_ids(); ++i) {
    tensorflow::Node* node = graph.FindNodeId(i);
    if (node->IsNextIteration()) {
      VLOG(2) << "num outputs before: " << node->num_outputs();
      for (auto edge : node->out_edges()) {
        VLOG(2) << "edge dst id:" << edge->dst()->id();
        nextiteration_edges.push_back(
            std::make_pair(edge->src()->id(), edge->dst()->id()));
        VLOG(2) << "Removing nextInteration edge: src->dst"
                << edge->src()->name() << " -> " << edge->dst()->name();
        graph.RemoveEdge(edge);
      }
    }
  }
}

tensorflow::Status SegmentGraphForLoops(
    const tensorflow::GraphDef& gdef,
    const std::function<bool(const tensorflow::Node*,
                             const std::set<std::string>&,
                             const std::set<std::string>&,
                             const std::set<std::string>&)>& candidate_fn,
    SegmentNodesVector* segments, std::set<string>& exclude_list,
    std::set<string>& loop_breaking_ops, std::set<std::string>* op_whitelist,
    std::set<std::string>* op_cpu, std::set<std::string>* op_inferentia) {
  // Create a Graph representation of the GraphDef.
  tensorflow::FunctionLibraryDefinition flib(tensorflow::OpRegistry::Global(),
                                             gdef.library());
  tensorflow::Graph graph(flib);
  TF_RETURN_IF_ERROR(tensorflow::ConvertGraphDefToGraph(
      tensorflow::GraphConstructorOptions(), gdef, &graph));

  // For graphs with loops, removing NextIteration out edge to break the
  // loop. This is only done for segmentation.
  RemoveCycleEdge(graph);

  tensorflow::kaena::segment::SegmentNodesVector loop_segments;
  GetLoopSegments(graph, op_whitelist, op_cpu, op_inferentia, &loop_segments,
                  exclude_list, loop_breaking_ops, candidate_fn);

  VLOG(2) << " Number of Loop segments: " << loop_segments.size();

  segments->insert(segments->end(), loop_segments.begin(), loop_segments.end());
  return tensorflow::Status::OK();
}

tensorflow::Status SegmentGraph(
    const tensorflow::GraphDef& gdef,
    const std::function<bool(const tensorflow::Node*,
                             const std::set<std::string>&,
                             const std::set<std::string>&,
                             const std::set<std::string>&)>& candidate_fn,
    SegmentOptions& options, SegmentNodesVector* segments,
    std::set<std::string>* op_whitelist,
    std::set<std::string>* op_cpu,
    std::set<std::string>* op_inferentia) {
  // Create a Graph representation of the GraphDef.
  tensorflow::FunctionLibraryDefinition flib(tensorflow::OpRegistry::Global(),
                                             gdef.library());
  tensorflow::Graph graph(flib);
  TF_RETURN_IF_ERROR(tensorflow::ConvertGraphDefToGraph(
      tensorflow::GraphConstructorOptions(), gdef, &graph));

  // For graphs with loops, removing NextIteration out edge to break the
  // loop. This is only done for segmentation.
  RemoveCycleEdge(graph);

  // The segmentation algorithm below visits nodes in reverse
  // topological order and attempts to merge nodes along output
  // edges. That means that subgraphs grow from the output-side of the
  // network towards the inputs. In general this is not guaranteed to
  // produce a globally optimal segmentation. In the future if we have
  // a measure of how beneficial it is to include a given node in a
  // EIA subgraph then we can revisit this algorithm to take advantage
  // of that information.
  std::vector<tensorflow::Node*> order;
  tensorflow::GetPostOrder(graph, &order);

  // Use a union-find to collect the nodes that belong to the same
  // segment. A node value of nullptr indicates that the node is not a
  // candidate for EIA.
  std::vector<UnionFind<tensorflow::Node*>> node_segments;
  for (int i = 0; i < graph.num_node_ids(); ++i) {
    tensorflow::Node* node = graph.FindNodeId(i);
    if (options.exclude_node_list.count(node->name()) != 0 ||
        !candidate_fn(node, *op_whitelist, *op_cpu, *op_inferentia)) {
      node = nullptr;
    }
    node_segments.emplace_back(node);
  }

  for (const tensorflow::Node* node : order) {
    // All output nodes of 'node' have been visited...
    VLOG(2) << "Trying node " << node->name() << " id=" << node->id()
            << " op= " << node->op_def().name();

    // 'node' must be a EIA candidate...
    if (node_segments[node->id()].Value() == nullptr) {
      VLOG(2) << "... not a EIA candidate";
      continue;
    }

    // Contract output edges to combine 'node' with output
    // nodes. Iterate since combining two nodes may unblock other
    // combining.
    while (true) {
      std::set<const tensorflow::Edge*> contract_edges;
      for (const tensorflow::Edge* out_edge : node->out_edges()) {
        VLOG(2) << "... out node " << out_edge->dst()->name() << " ( "
                << out_edge->dst()->id() << " <- " << node->id() << " )"
                << " ControlEdge: " << out_edge->IsControlEdge();

        // Out node must be EIA candidate...
        if (node_segments[out_edge->dst()->id()].Value() == nullptr) {
          VLOG(2) << "... ... not a EIA candidate";
          continue;
        }

        if (CanContractEdge(out_edge, graph)) {
          VLOG(2) << "... ... can contract";
          contract_edges.insert(out_edge);
        } else {
          VLOG(2) << "... ... cannot contract, would form cycle";
          VLOG(2) << " src: " << out_edge->src()->name()
                  << " dst: " << out_edge->dst()->name();
        }
      }

      if (contract_edges.empty()) {
        break;
      }

      // Contract edges and collect the adjacent nodes into the same
      // segment/subgraph.
      while (!contract_edges.empty()) {
        const tensorflow::Edge* contract_edge = *contract_edges.begin();
        const tensorflow::Node* src = contract_edge->src();
        const tensorflow::Node* dst = contract_edge->dst();

        VLOG(2) << "Merge " << src->name() << " <- " << dst->name() << " ("
                << src->id() << " <- " << dst->id();
        node_segments[src->id()].Merge(&node_segments[dst->id()]);

        // Contracting the edge leaves disconnected graph edges.
        // Remove these from the graph and from 'contract_edges' so we
        // don't visit them again.
        tensorflow::Edge* e = const_cast<tensorflow::Edge*>(contract_edge);
        std::vector<const tensorflow::Edge*> remove_edges;
        ContractEdge(e, &graph, &remove_edges);

        for (const tensorflow::Edge* r : remove_edges) {
          contract_edges.erase(r);
          graph.RemoveEdge(r);
        }
      }
    }
  }

  // Collect the segments/subgraphs. Each subgraph is represented by a
  // set of the names of the nodes in that subgraph.
  std::unordered_map<string, std::set<string>> sg_map;
  for (auto& u : node_segments) {
    if ((u.Value() != nullptr) && (u.ParentValue() != nullptr)) {
      sg_map[u.ParentValue()->name()].insert(u.Value()->name());
    }
  }

  // Convert the segments into the expected return format
  for (const auto& itr : sg_map) {
    const auto& segment_node_names = itr.second;
    if (VLOG_IS_ON(1)) {
      string s;
      for (const auto& name : segment_node_names) {
        s += " " + name;
      }
      VLOG(1) << "Segment " << segments->size() << ":" << s;
    }

    // Don't use small segments.
    if (static_cast<int>(segment_node_names.size()) <
        options.minimum_segment_size) {
      VLOG(1) << "Segment " << segments->size() << " has only "
              << segment_node_names.size() << " nodes, dropping";
      continue;
    }

    segments->emplace_back(segment_node_names);
  }

  return tensorflow::Status::OK();
}

}  // namespace segment
}  // namespace kaena
}  // namespace tensorflow
