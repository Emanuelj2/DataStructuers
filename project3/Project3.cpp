#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <limits>
#include <queue>
#include <functional>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cctype>
#include <map>
#include <set>

using namespace std;

unordered_map<string, string> airportscity, airportsstate; // airport's city, airport's state
unordered_map<string, set<string>> statesairports;         // state's airports

class Edge
{
public:
  string destination;
  double distance;
  double cost;
  Edge(const string &dest, double dist, double c) : destination(dest), distance(dist), cost(c) {}
};

enum class WeightType : int
{
  DISTANCE = 0,
  COST,
  WEIGHTEDSUM
};

class Weights
{
public:
  double distances;
  double costs;
  Weights() : distances(0), costs(0) {}
  Weights(double &d, double &c) : distances(d), costs(c) {}
};

class Graph
{
public:
  unordered_map<string, list<Edge>> adjList;
  set<string> vertices; //set will automatically delete duplications 
  unordered_map<string, int> inbounds, outbounds; //inbound in to the aitport, outboutd out of the airport 

  void addEdge(const string &origin, const string &destination, double distance, double cost)
  {
    adjList[origin].emplace_back(Edge(destination, distance, cost));
    vertices.insert(origin);
    vertices.insert(destination);
    if (inbounds.count(destination))
    {
      inbounds[destination]++; //if they exist 
    }
    else
    {
      inbounds.emplace(destination, 1); //if they dont exist 
    }

    if (outbounds.count(origin))//check if it exist 
    {
      outbounds[origin]++; //incrament if it exist
    }
    else
    {
      outbounds.emplace(origin, 1); //if they dont exist 
    }
  }

  void printGraph() const
  {
    for (const auto &pair : adjList)
    {
      cout << pair.first << ": ";
      for (const auto &edge : pair.second)
      {
        cout << "(" << edge.destination << ", " << edge.distance << ", " << edge.cost << ") ";
      }
      cout << endl;
    }
  }
};

class uGraph //undirected graph 
{
public:
  unordered_map<string, list<Edge>> adjList;
  map<tuple<string, string>, double> edges; //string1 orgin, string2 destinstion , double dist or cost

  void addEdge(const string &origin, const string &destination, double distance, double cost)
  {
    for (auto it = adjList[origin].begin(); it != adjList[origin].end(); ++it) //this mmeans that it is in the list
    {
      if (it->destination == destination)
      {
        return;
      }
    }
    adjList[origin].emplace_back(Edge(destination, distance, cost)); //this places the orgin to edge to the back of the list
    edges[{origin, destination}] = cost; // ignore distance

    for (auto it = adjList[destination].begin(); it != adjList[destination].end(); ++it)
    {
      if (it->destination == origin)
      {
        return;
      }
    }
    adjList[destination].emplace_back(Edge(origin, distance, cost));
    edges[{destination, origin}] = cost; // ignore distance
  }

  uGraph(Graph &dg, WeightType wt = WeightType::COST)
  {
    for (auto &p : dg.adjList)
    {
      double s2d = 0, d2s = 0, weight = 0;
      for (auto &e : p.second)
      {
        weight = s2d = (wt == WeightType::COST) ? e.cost : e.distance;
        for (auto it = dg.adjList[e.destination].begin(); it != dg.adjList[e.destination].end(); ++it)
        {
          if (it->destination == p.first)
          {
            d2s = (wt == WeightType::COST) ? it->cost : it->distance;
            weight = min(s2d, d2s);
          }
        }
        addEdge(p.first, e.destination, e.distance, weight);
      }
    }
  }

  void printuGraph() const
  {
    for (auto &pair : adjList)
    {
      cout << pair.first << ": ";
      for (auto &edge : pair.second)
      {
        cout << "(" << edge.destination << ", " << edge.distance << ", " << edge.cost << ") ";
      }
      cout << endl;
    }
  }
};

Weights cal_weights(Graph &graph, vector<string> &path)
{
  Weights ws = Weights();
  int n = path.size() - 1;

  for (size_t i = 0; i < n; i++)
  {
    for (auto &p : graph.adjList[path[i]])
      if (p.destination == path[i + 1])
      {
        ws.costs += p.cost;
        ws.distances += p.distance;
      }
  }

  return ws;
}

string trim(const string &str)
{
  auto it = find_if_not(str.begin(), str.end(), [](char c)
                        { return isspace(c); });
  auto end_it = find_if_not(str.rbegin(), str.rend(), [](char c)
                            { return isspace(c); })
                    .base();
  return string(it, end_it);
}

void getCityState(const string &citystate, string &city, string &state)
{
  city = state = "";
  bool citydone = false;
  for (char c : citystate)
  {
    if (c == ',' && !citydone)
    {
      citydone = !citydone;
      continue;
    }
    if (!citydone)
      city += c;
    else
      state += c;
  }
  city = trim(city);
  state = trim(state);
}

void parseCSV(const string &filename, Graph &graph)
{
  ifstream file(filename);
  if (!file.is_open())
  {
    throw runtime_error("Cannot open file");
  }

  string line;
  getline(file, line); // skip the headline
  vector<string> tokens;
  string token;
  bool inquotes = false;
  string oc, os, dc, ds;

  while (getline(file, line))
  {
    //
    // vector<string>: origin(0), destination(1), origin_city(2), destination_city(3), distance_str(4), cost_str(5);
    token = "";
    inquotes = false;
    tokens.clear();
    for (char c : line)
    {
      if ((c == ' ' || c == '\t') && !inquotes)
      {
        continue;
      }

      if (c == '"' || c == '\'')
      {
        inquotes = !inquotes;
        continue;
      }

      if (c == ',' && !inquotes)
      {
        tokens.push_back(token);
        token = "";
        continue;
      }
      token += c;
    }
    tokens.push_back(token); // push the last token

    double distance = stod(tokens[4]);
    double cost = stod(tokens[5]);

    getCityState(tokens[2], oc, os);
    getCityState(tokens[3], dc, ds);

    airportscity.emplace(tokens[0], oc);
    airportsstate.emplace(tokens[0], os);
    airportscity.emplace(tokens[1], dc);
    airportsstate.emplace(tokens[1], ds);

    if (statesairports.find(os) != statesairports.end())
    {
      statesairports[os].insert(tokens[0]);
    }
    else // key not found
    {
      statesairports.emplace(os, set<string>());
      statesairports[os].insert(tokens[0]);
    }

    if (statesairports.find(ds) != statesairports.end())
    {
      statesairports[ds].insert(tokens[1]);
    }
    else // key not found
    {
      statesairports.emplace(ds, set<string>());
      statesairports[ds].insert(tokens[1]);
    }

    graph.addEdge(tokens[0], tokens[1], distance, cost);
  }

  file.close();
}

pair<vector<string>, double> dijkstra(const Graph &graph, const string &start, const string &end, WeightType wt = WeightType::DISTANCE)
{
  unordered_map<string, double> weights;
  unordered_map<string, string> previous;
  priority_queue<pair<double, string>, vector<pair<double, string>>, greater<pair<double, string>>> pq; // smaller higher priority

  for (const auto &pair : graph.adjList)
  {
    weights[pair.first] = numeric_limits<double>::infinity();
  }
  weights[start] = 0;
  pq.push({0, start});

  while (!pq.empty())
  {
    double current_weight = pq.top().first;
    string current = pq.top().second;
    pq.pop();

    if (current == end)
      break;

    for (const Edge &edge : graph.adjList.at(current))
    {
      double weight = current_weight + (wt == WeightType::DISTANCE ? edge.distance : edge.cost);
      if (weight < weights[edge.destination])
      {
        weights[edge.destination] = weight;
        previous[edge.destination] = current;
        pq.push({weight, edge.destination});
      }
    }
  }

  vector<string> path;
  double total_cost = weights[end];
  for (string at = end; at != start; at = previous[at])
  {
    if (previous.find(at) == previous.end())
      return {{}, -1}; // No path found
    path.push_back(at);
  }
  path.push_back(start);
  reverse(path.begin(), path.end());

  return {path, total_cost};
}

void one2all(Graph &graph, string &origin, string &state)
{
  for (auto &destination : statesairports[state])
  {
    auto result = dijkstra(graph, origin, destination);
    if (result.second != -1)
    {
      cout << "Shortest route from " << origin << " to " << destination << ": ";
      for (const string &airport : result.first)
      {
        cout << airport << " ";
      }
      Weights ws = cal_weights(graph, result.first);
      cout << ". The length is " << result.second;
      cout << ". The cost is " << ws.costs << endl;
    }
    else
    {
      cout << "No route found from " << origin << " to " << destination << "." << endl;
    }
  }
}

vector<string> shortestPathWithIntermediates(Graph &graph, const string &start, const string &end, int intermediateCount, WeightType wt = WeightType::DISTANCE)
{
  struct State
  {
    string node;
    vector<string> path;
    double weight;
    int visitedCount;
  };

  auto compare = [](const State &a, const State &b)
  {
    return a.weight > b.weight;
  };

  priority_queue<State, vector<State>, decltype(compare)> pq(compare);
  pq.push({start, {start}, 0.0, 0});

  while (!pq.empty())
  {
    State current = pq.top();
    pq.pop();

    if (current.node == end && current.visitedCount == intermediateCount)
    {
      return current.path;
    }

    if (current.visitedCount > intermediateCount)
    {
      continue;
    }

    for (const auto &edge : graph.adjList[current.node])
    {
      string nextNode = edge.destination;
      double edgeWeight = (wt == WeightType::DISTANCE ? edge.distance : edge.cost);

      if (find(current.path.begin(), current.path.end(), nextNode) == current.path.end())
      {
        vector<string> newPath = current.path;
        newPath.push_back(nextNode);
        int newVisitedCount = (nextNode != end) ? current.visitedCount + 1 : current.visitedCount;
        pq.push({nextNode, newPath, current.weight + edgeWeight, newVisitedCount});
      }
    }
  }

  return {}; // No path found
}

//---
void dfs(uGraph &ug, const string &node, unordered_set<string> &visited, vector<string> &component)
{
  visited.insert(node);
  component.push_back(node);

  for (const auto &neighbor : ug.adjList[node])
  {
    if (visited.find(neighbor.destination) == visited.end())
    {
      dfs(ug, neighbor.destination, visited, component);
    }
  }
}

vector<vector<string>> findConnectedComponents(uGraph &ug)
{
  vector<vector<string>> connectedComponents;
  unordered_set<string> visited;

  for (const auto &entry : ug.adjList)
  {
    const string &node = entry.first;
    if (visited.find(node) == visited.end())
    {
      vector<string> component;
      dfs(ug, node, visited, component);
      connectedComponents.push_back(component);
    }
  }

  return connectedComponents;
}

vector<pair<string, string>> findMST(uGraph &ug, const vector<string> &component, double &totalCost)
{
  vector<pair<string, string>> mstEdges;
  unordered_set<string> visited;
  unordered_map<string, double> key;
  unordered_map<string, string> parent;

  // Initialize key values and parent
  for (const auto &node : component)
  {
    key[node] = numeric_limits<double>::infinity();
    parent[node] = "";
  }

  // Use a priority queue to get the minimum key value
  using pq_element = pair<double, string>;
  priority_queue<pq_element, vector<pq_element>, greater<pq_element>> pq;

  // Start with the first node in the component
  string start = component[0];
  key[start] = 0;
  pq.push({0, start});
  totalCost = 0;
  while (!pq.empty())
  {
    string u = pq.top().second;
    pq.pop();

    if (visited.find(u) != visited.end())
      continue;

    visited.insert(u);

    // Add edge to MST if it's not the start node
    if (parent[u] != "")
    {
      mstEdges.push_back({parent[u], u});
      totalCost += ug.edges[{parent[u], u}];
    }

    // Update key values of adjacent vertices
    for (const auto &neighbor : ug.adjList[u])
    {
      string v = neighbor.destination;
      double weight = neighbor.cost;

      if (visited.find(v) == visited.end() && weight < key[v])
      {
        parent[v] = u;
        key[v] = weight;
        pq.push({key[v], v});
      }
    }
  }

  return mstEdges;
}

class UnionFind
{
private:
  unordered_map<string, string> parent;
  unordered_map<string, int> rank;

public:
  void makeSet(const string &item)
  {
    if (parent.find(item) == parent.end())
    {
      parent[item] = item;
      rank[item] = 0;
    }
  }

  string find(const string &item)
  {
    if (parent[item] != item)
    {
      parent[item] = find(parent[item]);
    }
    return parent[item];
  }

  void unionSets(const string &x, const string &y)
  {
    string xRoot = find(x);
    string yRoot = find(y);

    if (xRoot == yRoot)
      return;

    if (rank[xRoot] < rank[yRoot])
    {
      parent[xRoot] = yRoot;
    }
    else if (rank[xRoot] > rank[yRoot])
    {
      parent[yRoot] = xRoot;
    }
    else
    {
      parent[yRoot] = xRoot;
      rank[xRoot]++;
    }
  }
};

vector<tuple<string, string, double>> findMST(uGraph &ug, vector<string> &component)
{
  vector<tuple<string, string, double>> mstEdges;
  UnionFind uf;

  // Initialize UnionFind for the component
  for (const auto &node : component)
  {
    uf.makeSet(node);
  }

  // Sort edges by weight
  vector<tuple<string, string, double>> componentEdges;
  for (const auto &edge : ug.edges)
  {
    const auto &[e, weight] = edge;
    if (find(component.begin(), component.end(), get<0>(e)) != component.end() &&
        find(component.begin(), component.end(), get<1>(e)) != component.end())
    {
      componentEdges.emplace_back(get<0>(e), get<1>(e), weight);
    }
  }
  sort(componentEdges.begin(), componentEdges.end(),
       [](const auto &a, const auto &b)
       { return get<2>(a) < get<2>(b); });

  // Kruskal's algorithm
  for (const auto &edge : componentEdges)
  {
    const auto &[from, to, weight] = edge;
    if (uf.find(from) != uf.find(to))
    {
      uf.unionSets(from, to);
      mstEdges.push_back(edge);
    }
  }

  return mstEdges;
}

int main()
{
  // Create and print an empty graph
  cout << "\n===Task 0: Build an empty graph for airports==============" << endl;
  Graph emptyGraph;
  emptyGraph.printGraph();

  // Task 1: Build the graph of airports
  cout << "\n===Task 1: Build the graph of airports==============" << endl;
  // 1.1 parse the airport csv file, infill the empty graph, then print it
  Graph g;
  parseCSV("airports.csv", g);
  g.printGraph();
  // 1.2 print all the airports, their cities and states
    cout << "================== airport and city ======================" << endl;
  for(auto a: airportscity){
    cout << a.first << ": " << a.second << endl;
  }
  cout << "================== airport and state ======================" << endl;
  for(auto a: airportsstate){
    cout << a.first << ": " << a.second << endl;
  }
  // airport (city, state)

  // 1.3 print all the airports in each state
  cout << "================== airports in each state ======================" << endl;
  for (const auto &entry: statesairports){
    cout << entry.first << ": ";
    for(const auto &airport: entry.second){
      cout << airport << " ";
    }
    cout << endl;
  }

  // Task 2: Shortest path between two airports using Dijkstra
  cout << "\n===Task 2: Shortest path between two airports using Dijkstra============" << endl;
  cout <<endl;
  // 2.1 print at least 2 pairs of airports WITH shortest paths between them
  string start1, dest1, start2, dest2;
  start1 = "IDA", dest1 = "MIA";
  auto res1 = dijkstra(g, start1, dest1, WeightType::DISTANCE);
  cout <<"Shortest route from "<<  start1 << " to " << dest1 << ": ";
  for(auto a: res1.first){
    cout << a << "->";
  }
  auto total1 = cal_weights(g, res1.first);
  cout << "Cost: $" << total1.costs << " Distance: " << total1.distances << endl;

  start2 = "SFO", dest2 = "MIA";
  auto res2 = dijkstra(g, start2, dest2, WeightType::DISTANCE);
  cout << "Shortest rout from " << start2 << " to " << dest2 << ": ";
  for(auto a: res2.first){
    cout << a << "->";
  }
  auto total2 = cal_weights(g, res2.first);
  cout << "Cost: $" << total2.costs << " Distance: " << total2.distances << endl;

  // 2.2 print at least 2 pairs of airports WITHOUT shortest paths between them
  cout <<endl;
  string start3 = "ACT", dest3 = "MIA";
  cout << "Shortest rout from " << start3 << " to " << dest3 <<": ";
  try{
    auto res3 = dijkstra(g, start3, dest3);
  }
  catch(const exception& e){
    cout << "none" << endl << e.what() << endl;
  }
  
  string start4 = "ACT", dest4 = "PIT";
  cout << "Shortest rout from " << start4 << " to " << dest4 << ": ";
  try{
    auto res4 = dijkstra(g, start4, dest4);
  }
  catch(const exception &e){
    cout << "none" << endl << e.what() << endl;
  }
  
  // Task 3: Shortest paths from one airport to all the airports in a state
  cout << "\n===Task 3: Shortest paths from one airport to all the airports in a state======" << endl;
  // 3.1 print at least 2 airports to all the airports in two NON-HOME states
  cout << " ==============NON-HOME STATES 'TPA' to 'TX'==============" <<endl;
  string from1 = "TPA";
  string nonHome1 = "TX";
  one2all(g, from1, nonHome1);
  cout << endl;
  cout << " ==============NON-HOME STATES 'MIA' to 'MA'==============" <<endl; 
  string from2 = "MIA";
  string nonHome2 = "MA";
  one2all(g, from2, nonHome2);
  cout << endl;
  // 3.2 print at least 2 airports to all the airports in two HOME states
  cout << " ==============HOME STATES 1==============" <<endl;
  string homestate1 = "FL";
  one2all(g, from1, homestate1);
  cout << endl;
  cout << " ==============HOME STATES 2==============" <<endl;
  string homestate2 = "FL";
  one2all(g, from2, homestate2);
  // Task 4: Shortest path between two airports with specified number of stops
  cout << "\n===Task 4: Shortest path between two airports with specified number of stops====" << endl;

   
  // 4.1 print at least two flights with 2 stops
string start41 = "SFO";
string dest41 = "EWR";
int stops41 = 2;
auto shortestpath41 = shortestPathWithIntermediates(g, start41, dest41, stops41, WeightType::DISTANCE);
cout << "Shortest route from " << start41 << " to " << dest41 << " with " << stops41 << " stops: ";
for (auto a : shortestpath41) {
    cout << a << "->";
}
cout << endl;
auto total41 = cal_weights(g, shortestpath41);
cout << "Cost: $" << total41.costs << " Distance: " << total41.distances << endl;

string start412 = "MIA";
string dest412 = "MDW";
auto shortestpath412 = shortestPathWithIntermediates(g, start412, dest412, 2, WeightType::DISTANCE);
cout << "Shortest route from " << start412 << " to " << dest412 << " with 2 stops: ";
for (auto a : shortestpath412) {
  cout << a << "->"; 
}
cout << endl;
auto total412 = cal_weights(g, shortestpath412);
cout << "Cost: $" << total412.costs << " Distance: " << total412.distances << endl;

  // 4.2 print at least two flights with 5 stops

string start42 = "BOS";
string dest42 = "ORD";
int stops42 = 5;
auto shortestpath42 = shortestPathWithIntermediates(g, start42, dest42, stops42, WeightType::DISTANCE);
cout << "Shortest route from " << start42 << " to " << dest42 << " with " << stops42 << " stops: ";
for (auto a : shortestpath42) {
    cout << a << "->";
}
cout << endl;
auto total42 = cal_weights(g, shortestpath42);
cout << "Cost: $" << total42.costs << " Distance: " << total42.distances << endl;

string start422 = "AFW";
string dest422 = "EWR";
auto shortestpath422 = shortestPathWithIntermediates(g, start422, dest422, stops42, WeightType::DISTANCE);
cout << "Shortest route from " << start422 << " to " << dest422 << " with " << stops42 << " stops: ";
for(auto a:  shortestpath422){
  cout << a <<"->";
}
cout << endl;
auto total422 = cal_weights(g, shortestpath422);
cout << "Cost: $" << total422.costs << " Distance: " << total422.distances << endl;
  // 4.1 print at least one fail case for 4.1
string startFail41 = "SGF";
string destFail41 = "ACT"; // Assuming XXX is an invalid destination
auto shortestpathFail41 = shortestPathWithIntermediates(g, startFail41, destFail41, 2, WeightType::DISTANCE);
if (shortestpathFail41.empty()) {
    cout << "No route found from " << startFail41 << " to " << destFail41 << " with 2 stops." << endl;
} else {
    cout << "Shortest route from " << startFail41 << " to " << destFail41 << " with 2 stops: ";
    for (auto a : shortestpathFail41) {
        cout << a << "->";
    }
    cout << endl;
    auto totalFail41 = cal_weights(g, shortestpathFail41);
    cout << "Cost: $" << totalFail41.costs << " Distance: " << totalFail41.distances << endl;
}

  // 4.2 print at least one fail case for 4.2
string startFail42 = "EAT";
string destFail42 = "YKM"; // Assuming YYY is an invalid destination
auto shortestpathFail42 = shortestPathWithIntermediates(g, startFail42, destFail42, 5, WeightType::DISTANCE);
if (shortestpathFail42.empty()) {
    cout << "No route found from " << startFail42 << " to " << destFail42 << " with 5 stops." << endl;
} else {
    cout << "Shortest route from " << startFail42 << " to " << destFail42 << " with 5 stops: ";
    for (auto a : shortestpathFail42) {
        cout << a << "->";
    }
    cout << endl;
    auto totalFail42 = cal_weights(g, shortestpathFail42);
    cout << "Cost: $" << totalFail42.costs << " Distance: " << totalFail42.distances << endl;
}
  // Task 5: Total direct flight connections to each airport
  cout << "\n===Task 5: Total direct flight connections to each airport====" << endl;
  unordered_map <string, int> airportConnections;
  for(const auto &entry: g.adjList){
    for(const auto &flight: entry.second){
      airportConnections[flight.destination]++;
    }
  }
  for(const auto& airport: airportConnections){
    cout << "Airport " << airport.first << " has " << airport.second << " direct connections." << endl;
  }
  // TODO 6: Task 6: Convert the directed graph into an undirected graph
  cout << "\n===Task 6: Convert the directed graph into an undirected graph====" << endl;
  uGraph ug(g);
  ug.printuGraph();

  //Task 7: Generate Minimal Spanning Trees utilizing Prim's algorithm
  cout << "\n===Task 7: Generate Minimal Spanning Trees utilizing Prim's algorithm====" << endl;
  // 7.1 find all connected components
  cout << "====================all connected components======================"<< endl;
  
  auto coms = findConnectedComponents(ug); 
  for(auto c: coms){
    for(auto a: c){
      cout << a << " ";
    }
    cout << "\n *";
  }

  // 7.2 find the MST for each connected component
  cout  <<"===============MST for each connected component==================="<< endl;
  int ic = 0;
  for(auto c : coms){
    double tc7 = 0;
    ic++;
    auto r72 = findMST(ug, c , tc7);
    cout << "The MST of connected component " << ic << ": " <<endl;
    for(auto e: r72){
      cout << e.first << "---" << e.second << "(" << ug.edges[e] << ")" << endl;
    }
    cout <<"Total weights: " << tc7 << endl;
  }
  // TODO 8: Task 8: Generate Minimal Spanning Trees utilizing Kruskal's algorithm
  cout << "\n===Task 8: Generate Minimal Spanning Trees utilizing Kruskal's algorithm====" << endl;
  // 8.2 find the MST for each connected component
  int ic8 = 0;
  for(auto c: coms){
    double tc8 = 0;
    ic8++;
    auto r8 = findMST(ug, c);
    cout << "The MST of connected components " << ic8 << ": " << endl;
    for(auto e: r8){
      cout << get<0>(e) << "---" << get<1>(e)  << "("<< get<2>(e) << ")" << endl;
      ic8 += get<2>(e);
    }
    cout << "Total weights: " << ic8 << endl;
  }

  return 0;
}
