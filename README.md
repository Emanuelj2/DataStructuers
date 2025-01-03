# Graph-Based Airport Connectivity and Flight Route Optimization System

## Overview

The **Graph-Based Airport Connectivity and Flight Route Optimization System** is a software application designed to enhance the efficiency of airport connectivity and optimize flight routes. By leveraging advanced graph algorithms and data structures, this system aims to improve route planning, reduce travel costs, and provide insights into airport networks worldwide.

---

## Features

- **Graph Representation**: Models airports as nodes and flight routes as weighted edges, capturing real-world connectivity.
- **Optimal Pathfinding**: Implements algorithms such as Dijkstra's and A* for shortest path calculations.
- **Route Optimization**: Offers flight route optimization based on distance, cost, and time.
- **Data Analytics**: Analyzes connectivity metrics like centrality, degree distribution, and bottlenecks.

---

## Technologies Used

- **Programming Language**: C++
- **Graph Libraries**: 

---

## System Requirements

- C++ compiler (GCC or MSVC)
- Modern terminal or IDE for C++

---

## Installation

### Prerequisites
1. Install a C++ compiler:
    ```bash
    sudo apt-get install g++
    ```
2. Install CMake:
    ```bash
    sudo apt-get install cmake
    ```
3. Install Boost library:
    ```bash
    sudo apt-get install libboost-all-dev
    ```

### Steps
1. Clone the repository:

2. Navigate to the project directory:
    ```bash
    cd Graph-Based-Airport-System
    ```
3. Build the project:
    ```bash
    mkdir build && cd build
    cmake ..
    make
    ```
4. Run the application:
    ```bash
    g++ -std=c++11 -o test Graph-Based-Airport-System.cpp
    ./test
    ```
---

## Algorithms

- **Dijkstra's Algorithm**: Finds the shortest path between two airports.
- **Bellman-Ford Algorithm**: Computes shortest paths while handling negative weights.
- **Floyd-Warshall Algorithm**: Calculates shortest paths for all pairs of nodes.
- **Minimum Spanning Tree**: Optimizes overall connectivity using Prim's or Kruskal's algorithms.


