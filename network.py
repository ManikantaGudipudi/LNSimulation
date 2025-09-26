#!/usr/bin/env python3
"""
Lightning Network — Sender-View Routing Comparison
Compares routing performance with and without information sharing.

Two Simulation Modes:
1. No Info Sharing: Only sender/receiver knowledge, no intermediate node sharing
2. With Info Sharing: Sender/receiver + willing intermediate nodes share information

Both modes use edge pruning to avoid re-trying failed paths.
"""

from __future__ import annotations

import json
import random
import time
from typing import Dict, List, Tuple, Optional, Iterable, NamedTuple
from dataclasses import dataclass
from enum import Enum

import networkx as nx


# =============================================================================
# Data Structures and Enums
# =============================================================================

class SimulationMode(Enum):
    """Simulation modes for information sharing."""
    NO_INFO_SHARING = "no_info_sharing"
    WITH_INFO_SHARING = "with_info_sharing"

@dataclass
class SimulationResult:
    """Results from a single simulation run."""
    mode: SimulationMode
    success: bool
    execution_time: float
    paths_checked: int
    final_path: Optional[List[str]] = None
    failure_reason: Optional[str] = None

@dataclass
class ComparisonResult:
    """Results comparing two simulation modes."""
    no_sharing: SimulationResult
    with_sharing: SimulationResult
    time_saved: float
    time_saved_percentage: float
    paths_saved: int
    paths_saved_percentage: float


# =============================================================================
# Base: Data loading and graph construction
# =============================================================================

class LightningNetworkBase:
    """
    Loads LN JSON and builds:
      - G (nx.Graph): undirected active channels with 'capacity' (satoshis)
      - G_dir (nx.DiGraph): directed balances by splitting each undirected capacity
    """

    def __init__(self, json_file: str = "LNdata.json", seed: Optional[int] = 42):
        self.json_file = json_file
        self.channels_data: List[dict] = []
        self.G: nx.Graph = nx.Graph()
        self.G_dir: nx.DiGraph = nx.DiGraph()
        self._rng = random.Random(seed)

    # ---------- I/O ----------

    def load_data(self) -> None:
        t0 = time.time()
        with open(self.json_file, "r") as f:
            data = json.load(f)
        self.channels_data = data.get("channels", [])
        print(f"Loaded {len(self.channels_data)} total channels  ({time.time()-t0:.2f}s)")

    # ---------- Graphs ----------

    def build_undirected(self) -> None:
        """Keep only active channels; store 'capacity' (sum over parallel channels)."""
        t0 = time.time()
        self.G.clear()
        for ch in self.channels_data:
            if not ch.get("active", False):
                continue
            u = ch.get("source")
            v = ch.get("destination")
            cap = ch.get("satoshis", 0)
            if not u or not v or cap <= 0:
                continue
            if self.G.has_edge(u, v):
                self.G[u][v]["capacity"] += float(cap)
            else:
                self.G.add_edge(u, v, capacity=float(cap))
        print(f"Undirected: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} active edges  ({time.time()-t0:.2f}s)")

    def build_directed_from_undirected(self) -> None:
        """
        Split each undirected capacity into two directed balances (random but seedable).
        NOTE: This is ground truth and is never modified during the simulation.
        """
        t0 = time.time()
        self.G_dir.clear()
        self.G_dir.add_nodes_from(self.G.nodes(data=True))
        for u, v, data in self.G.edges(data=True):
            cap = int(data.get("capacity", 0))
            if cap <= 1:
                continue
            cap_uv = self._rng.randint(1, cap - 1)  # u->v
            cap_vu = cap - cap_uv                   # v->u
            self.G_dir.add_edge(u, v, capacity=float(cap_uv))
            self.G_dir.add_edge(v, u, capacity=float(cap_vu))
        print(f"Directed (balances): {self.G_dir.number_of_nodes()} nodes, {self.G_dir.number_of_edges()} edges  ({time.time()-t0:.2f}s)")


# =============================================================================
# Sender-View Simulation (No Rebalancing) + Edge Pruning
# =============================================================================

class SenderViewSim(LightningNetworkBase):
    """
    Edge-smart sender simulation with configurable information sharing:
      - Routing over a mutable copy G_route (we prune edges on failures)
      - Ground-truth directed balances G_dir (immutable)
      - Sender initially knows only their own edges (not receiver's)
      - Receiver and intermediate nodes can share information based on willingness
      - Supports both no-info-sharing and with-info-sharing modes
      - Pre-filter s,t to ensure shortest path is long enough
      - No-info-sharing mode: prune the *first failing hop* and restart path enumeration
      - With-info-sharing mode: prune ALL insufficient edges we know about after sharing
    """

    def __init__(self, json_file: str = "LNdata.json", seed: Optional[int] = 42, 
                 mode: SimulationMode = SimulationMode.WITH_INFO_SHARING, verbose: bool = True):
        super().__init__(json_file=json_file, seed=seed)
        self.mode = mode
        self.verbose = verbose
        # sender_view[node] = {"out": {nbr: cap}, "in": {nbr: cap}}
        self.sender_view: Dict[str, Dict[str, Dict[str, float]]] = {}

    # -------------------- Knowledge / Checks --------------------

    def _init_sender_view_for(self, node: str) -> None:
        """
        Initialize sender's knowledge for 'node' from TRUE balances (G_dir).
        Safe to call multiple times (idempotent).
        """
        out_view = {nbr: float(self.G_dir[node][nbr]["capacity"])
                    for nbr in self.G_dir.successors(node)}
        in_view = {nbr: float(self.G_dir[nbr][node]["capacity"])
                   for nbr in self.G_dir.predecessors(node)}
        self.sender_view[node] = {"out": out_view, "in": in_view}

    @staticmethod
    def _true_path_feasible(G_dir: nx.DiGraph, path: List[str], amount: float) -> bool:
        """
        Ground-truth check on directed balances: every edge u->v along 'path' must have cap >= amount.
        """
        for u, v in zip(path[:-1], path[1:]):
            if not G_dir.has_edge(u, v):
                return False
            if G_dir[u][v]["capacity"] < amount:
                return False
        return True

    @staticmethod
    def _first_failing_true_edge(
        G_dir: nx.DiGraph,
        path: List[str],
        amount: float
    ) -> Optional[Tuple[str, str]]:
        """
        Return the first (u, v) along 'path' where TRUE directed capacity < amount.
        """
        for u, v in zip(path[:-1], path[1:]):
            if (not G_dir.has_edge(u, v)) or (G_dir[u][v]["capacity"] < amount):
                return (u, v)
        return None

    def _first_failing_hop_by_view(
        self,
        path: List[str],
        amount: float,
        G_route: nx.Graph,
    ) -> Optional[Tuple[str, str, str]]:
        """
        Find the *first* hop (u,v) along 'path' that fails under the sender's current knowledge.
        Returns (u, v, reason) where reason ∈ {'directed_src', 'directed_dst', 'undirected'}.
        If all hops pass, returns None.
        """
        for u, v in zip(path[:-1], path[1:]):
            # Directed known from u's OUT?
            if (u in self.sender_view) and (v in self.sender_view[u]["out"]):
                if self.sender_view[u]["out"][v] < amount:
                    return (u, v, "directed_src")
                continue
            # Directed known from v's IN?
            if (v in self.sender_view) and (u in self.sender_view[v]["in"]):
                if self.sender_view[v]["in"][u] < amount:
                    return (u, v, "directed_dst")
                continue
            # Fallback to undirected
            if not G_route.has_edge(u, v) or G_route[u][v]["capacity"] < amount:
                return (u, v, "undirected")
        return None

    def _find_all_insufficient_edges_after_sharing(
        self,
        path: List[str],
        amount: float,
        G_route: nx.Graph,
    ) -> List[Tuple[str, str, str]]:
        """
        Find ALL edges along 'path' that have insufficient capacity after information sharing.
        Returns list of (u, v, reason) tuples where reason ∈ {'directed_src', 'directed_dst', 'undirected'}.
        Only includes edges we have information about (not unknown edges).
        """
        insufficient_edges = []
        
        for u, v in zip(path[:-1], path[1:]):
            # Check if we have directed knowledge from u's OUT
            if (u in self.sender_view) and (v in self.sender_view[u]["out"]):
                if self.sender_view[u]["out"][v] < amount:
                    insufficient_edges.append((u, v, "directed_src"))
                continue  # Skip other checks for this edge
            
            # Check if we have directed knowledge from v's IN
            if (v in self.sender_view) and (u in self.sender_view[v]["in"]):
                if self.sender_view[v]["in"][u] < amount:
                    insufficient_edges.append((u, v, "directed_dst"))
                continue  # Skip other checks for this edge
            
            # For edges we don't have directed knowledge about, check undirected capacity
            # Only include if we know the undirected capacity is insufficient
            if G_route.has_edge(u, v):
                if G_route[u][v]["capacity"] < amount:
                    insufficient_edges.append((u, v, "undirected"))
            # Note: We don't include edges that don't exist in G_route as "insufficient"
            # because we don't have capacity information about them
        
        return insufficient_edges

    def _pre_prune_source_edges(self, G_route: nx.Graph, source: str, amount: float) -> None:
        """
        Pre-prune source outgoing edges with insufficient capacity before path finding.
        This replaces Phase 1 testing by removing known bad edges upfront.
        """
        if source not in self.sender_view:
            return
        
        edges_to_remove = []
        for neighbor in self.sender_view[source]["out"]:
            if self.sender_view[source]["out"][neighbor] < amount:
                edges_to_remove.append((source, neighbor))
        
        for u, v in edges_to_remove:
            if G_route.has_edge(u, v):
                G_route.remove_edge(u, v)
                if self.verbose:
                    print(f"Pre-pruned insufficient source edge: {u} -- {v} (capacity: {self.sender_view[source]['out'][v]})")
        
        if self.verbose and edges_to_remove:
            print(f"Pre-pruned {len(edges_to_remove)} insufficient source edges")

    # -------------------- Paths --------------------

    @staticmethod
    def _shortest_paths_generator(G: nx.Graph, s: str, t: str) -> Iterable[List[str]]:
        """
        Generate simple paths from s to t in increasing number of hops
        (networkx.shortest_simple_paths).
        """
        return nx.shortest_simple_paths(G, s, t)

    def pick_sender_receiver_with_min_distance(
        self,
        G_route: nx.Graph,
        min_path_nodes: int,
        max_tries: int = 300,
        verbose: bool = True,
    ) -> Tuple[str, str]:
        """
        Randomly sample (s,t) until the *shortest path length* in hops is >= (min_path_nodes-1).
        """
        assert min_path_nodes >= 2
        nodes = list(G_route.nodes())
        need_hops = min_path_nodes - 1
        tries = 0
        while tries < max_tries:
            tries += 1
            s, t = self._rng.sample(nodes, 2)
            try:
                d = nx.shortest_path_length(G_route, s, t)
                if d >= need_hops:
                    if verbose:
                        print(f"Chosen s,t after {tries} tries (shortest hops={d}, need≥{need_hops})")
                    return s, t
            except nx.NetworkXNoPath:
                continue
        if verbose:
            print(f"Could not find s,t with distance≥{need_hops} in {max_tries} tries; using last sample.")
        return s, t

    # -------------------- Structured Printing --------------------

    def _print_path_structured(
        self,
        path_idx: int,
        path: List[str],
        amount: float,
        G_view_for_undirected: nx.Graph,
        phase: str,
        show_nodes_line: bool = True,
        header: str = "Candidate Path",
    ) -> None:
        """
        Pretty-print a path with hop-by-hop details:
          u -> v | known_src? | known_dst? | dir_cap(used) | undir_cap | rule_used | pass?
        """
        hops = len(path) - 1
        if show_nodes_line:
            if len(path) > 8:
                compact_nodes = path[:2] + ["..."] + path[-2:]
            else:
                compact_nodes = path
            print(f"\n{header} #{path_idx} [{phase}]  (hops={hops}, nodes={len(path)})")
            print("Nodes:", "  →  ".join(compact_nodes))

        # Columns
        col_u = "u"
        col_v = "v"
        col_k_src = "known_src"
        col_k_dst = "known_dst"
        col_dir_used = "dir_cap_used"
        col_undir = "undir_cap"
        col_rule = "rule"
        col_ok = f">={int(amount)}?"

        widths = {
            col_u: 18,
            col_v: 18,
            col_k_src: 9,
            col_k_dst: 9,
            col_dir_used: 12,
            col_undir: 10,
            col_rule: 12,
            col_ok: 8,
        }

        def fmt(name, value):
            s = str(value)
            w = widths[name]
            return s[:w].ljust(w)

        header_line = (
            f"{fmt(col_u, col_u)} | "
            f"{fmt(col_v, col_v)} | "
            f"{fmt(col_k_src, col_k_src)} | "
            f"{fmt(col_k_dst, col_k_dst)} | "
            f"{fmt(col_dir_used, col_dir_used)} | "
            f"{fmt(col_undir, col_undir)} | "
            f"{fmt(col_rule, col_rule)} | "
            f"{fmt(col_ok, col_ok)}"
        )
        sep_line = "-" * len(header_line)
        print(sep_line)
        print(header_line)
        print(sep_line)

        for u, v in zip(path[:-1], path[1:]):
            known_src = (u in self.sender_view) and (v in self.sender_view[u]["out"])
            known_dst = (v in self.sender_view) and (u in self.sender_view[v]["in"])
            undir_cap = G_view_for_undirected[u][v]["capacity"] if G_view_for_undirected.has_edge(u, v) else 0.0

            rule = "undirected"
            dir_used = "—"
            passes = undir_cap >= amount

            if known_src:
                rule = "directed(src)"
                dir_val = self.sender_view[u]["out"][v]
                dir_used = int(dir_val)
                passes = dir_val >= amount
            elif known_dst:
                rule = "directed(dst)"
                dir_val = self.sender_view[v]["in"][u]
                dir_used = int(dir_val)
                passes = dir_val >= amount

            print(
                f"{fmt(col_u, u)} | "
                f"{fmt(col_v, v)} | "
                f"{fmt(col_k_src, 'yes' if known_src else 'no')} | "
                f"{fmt(col_k_dst, 'yes' if known_dst else 'no')} | "
                f"{fmt(col_dir_used, dir_used)} | "
                f"{fmt(col_undir, int(undir_cap))} | "
                f"{fmt(col_rule, rule)} | "
                f"{fmt(col_ok, 'YES' if passes else 'NO')}"
            )
        print(sep_line)

    # -------------------- Main loop with pruning --------------------

    def run_simulation(
        self,
        amount: float = 700000.0,
        p_willing: float = 0.6,
        min_path_nodes: int = 2,          # nodes per path (set per your experiments)
        max_paths_checked: int = 1000,    # across all restart rounds
        include_endpoints_in_share: bool = False,  # endpoints already known
        seed_sender_receiver: Optional[Tuple[str, str]] = None,
    ) -> SimulationResult:
        """
        Edge-pruning search with restarts, including pruning after TRUE directed failure.
        Returns structured results for comparison.
        """
        start_time = time.time()
        
        # Working routing graph that we prune
        G_route = self.G.copy()

        # Choose s,t with hop-distance prefilter on the *current* G_route
        if seed_sender_receiver:
            s, t = seed_sender_receiver
            if s == t:
                raise ValueError("Sender and receiver must be distinct.")
            if s not in G_route or t not in G_route:
                raise ValueError("Seeded sender/receiver not found in graph.")
        else:
            s, t = self.pick_sender_receiver_with_min_distance(
                G_route=G_route, min_path_nodes=min_path_nodes, max_tries=500
            )

        if self.verbose:
            print(f"\nSender:   {s}")
            print(f"Receiver: {t}")
            print(f"Mode: {self.mode.value}")

        # Initialize knowledge for SENDER ONLY initially
        self.sender_view.clear()
        self._init_sender_view_for(s)
        
        if self.verbose:
            print("\nInitialized sender's local view for: sender ONLY (receiver knowledge will come from sharing).")

        # Pre-prune source outgoing edges with insufficient capacity
        self._pre_prune_source_edges(G_route, s, amount)
        
        paths_used = 0  # total paths tested across restarts

        while paths_used < max_paths_checked:
            # If s,t disconnected now, we're done
            try:
                _ = nx.shortest_path_length(G_route, s, t)
            except nx.NetworkXNoPath:
                if self.verbose:
                    print("\nRouting graph became disconnected between s and t. Stopping.")
                execution_time = time.time() - start_time
                return SimulationResult(
                    mode=self.mode,
                    success=False,
                    execution_time=execution_time,
                    paths_checked=paths_used,
                    failure_reason="Graph disconnected"
                )

            # Build a fresh generator after any pruning
            try:
                gen = self._shortest_paths_generator(G_route, s, t)
            except nx.NetworkXNoPath:
                if self.verbose:
                    print("\nNo path between sender and receiver in the routing graph.")
                execution_time = time.time() - start_time
                return SimulationResult(
                    mode=self.mode,
                    success=False,
                    execution_time=execution_time,
                    paths_checked=paths_used,
                    failure_reason="No path found"
                )

            pruned_this_round = False

            for path in gen:
                if len(path) < min_path_nodes:
                    continue

                paths_used += 1
                if paths_used > max_paths_checked:
                    if self.verbose:
                        print("\nNo feasible route found within path budget.")
                    execution_time = time.time() - start_time
                    return SimulationResult(
                        mode=self.mode,
                        success=False,
                        execution_time=execution_time,
                        paths_checked=paths_used,
                        failure_reason="Path budget exceeded"
                    )

                # -------- Phase A: info sharing by willingness (only if mode allows) --------
                if self.mode == SimulationMode.WITH_INFO_SHARING:
                    willingness = {node: (self._rng.random() < p_willing) for node in path}
                    # Always include receiver in sharing since sender doesn't know receiver initially
                    nodes_to_share = path[1:]  # All nodes except sender (receiver + intermediates)
                    if self.verbose:
                        print("\nWillingness among nodes (excluding sender):")
                        for node in nodes_to_share:
                            flag = "willing" if willingness.get(node, False) else "not willing"
                            print(f"  {node}: {flag}")

                    for node in nodes_to_share:
                        if willingness.get(node, False):
                            self._init_sender_view_for(node)

                    # Re-check after shares
                    if self.verbose:
                        self._print_path_structured(
                            path_idx=paths_used,
                            path=path,
                            amount=amount,
                            G_view_for_undirected=G_route,
                            phase="A: after shares",
                            show_nodes_line=True,
                            header="Candidate Path",
                        )
                    
                    # Find and prune ALL insufficient edges we now know about
                    insufficient_edges = self._find_all_insufficient_edges_after_sharing(path, amount, G_route)
                    feasible_after_shares = (len(insufficient_edges) == 0)
                    if self.verbose:
                        print(f"Feasible after shares? {'YES' if feasible_after_shares else 'NO'}")
                        if insufficient_edges:
                            print(f"Found {len(insufficient_edges)} insufficient edges after sharing")

                    if not feasible_after_shares:
                        # PRUNE ALL insufficient edges we now know about
                        edges_pruned = 0
                        for u, v, reason in insufficient_edges:
                            if G_route.has_edge(u, v):
                                G_route.remove_edge(u, v)
                                edges_pruned += 1
                                if self.verbose:
                                    print(f"Pruned edge due to {reason}: {u} -- {v}")
                            else:
                                if self.verbose:
                                    print(f"No prune (edge missing): {u} -- {v} (reason={reason})")
                        
                        if self.verbose:
                            print(f"Total edges pruned after sharing: {edges_pruned}")
                        pruned_this_round = True
                        break  # restart generator
                else:
                    # No info sharing mode - no additional processing needed
                    if self.verbose:
                        print("No info sharing mode - using pre-pruned graph")

                # -------- Try the TRUE send on G_dir --------
                true_ok = self._true_path_feasible(self.G_dir, path, amount)
                if true_ok:
                    execution_time = time.time() - start_time
                    if self.verbose:
                        print("Actual send result: SUCCESS ✅")
                    return SimulationResult(
                        mode=self.mode,
                        success=True,
                        execution_time=execution_time,
                        paths_checked=paths_used,
                        final_path=path
                    )
                else:
                    # Locate and report the first failing TRUE edge (u → v)
                    failing_true = self._first_failing_true_edge(self.G_dir, path, amount)
                    if failing_true is None:
                        execution_time = time.time() - start_time
                        if self.verbose:
                            print("Actual send result: FAILURE ❌ (but could not isolate failing true edge)")
                        return SimulationResult(
                            mode=self.mode,
                            success=False,
                            execution_time=execution_time,
                            paths_checked=paths_used,
                            failure_reason="Could not isolate failing edge"
                        )
                    u, v = failing_true
                    # Visibility: was this edge known to the sender?
                    known_src = (u in self.sender_view) and (v in self.sender_view[u]["out"])
                    known_dst = (v in self.sender_view) and (u in self.sender_view[v]["in"])
                    visibility = "HIDDEN" if not (known_src or known_dst) else "KNOWN"
                    if self.verbose:
                        print(f"Actual send result: FAILURE ❌  True failing edge: {u} -> {v}  [{visibility}]")
                    # Prune the undirected edge and restart
                    if G_route.has_edge(u, v):
                        G_route.remove_edge(u, v)
                        if self.verbose:
                            print(f"Pruned edge due to TRUE failure: {u} -- {v}")
                    else:
                        if self.verbose:
                            print(f"No prune (edge missing in routing graph): {u} -- {v}")
                    pruned_this_round = True
                    break  # restart generator after pruning

            if not pruned_this_round:
                # Generator exhausted without any prune or success → nothing left
                if self.verbose:
                    print("\nExhausted candidate paths on current routing graph; stopping.")
                break

        execution_time = time.time() - start_time
        if self.verbose:
            print("\nNo feasible route found (or path budget exhausted).")
        return SimulationResult(
            mode=self.mode,
            success=False,
            execution_time=execution_time,
            paths_checked=paths_used,
            failure_reason="No feasible route found"
        )


# =============================================================================
# Comparison Runner
# =============================================================================

class SimulationComparison:
    """Runs and compares simulations with and without information sharing."""
    
    def __init__(self, json_file: str = "LNdata.json", seed: Optional[int] = 42):
        self.json_file = json_file
        self.seed = seed
        self.base_sim = None
        
    def _setup_base_simulation(self) -> None:
        """Load data and build graphs once for both simulations."""
        if self.base_sim is None:
            self.base_sim = LightningNetworkBase(json_file=self.json_file, seed=self.seed)
            self.base_sim.load_data()
            self.base_sim.build_undirected()
            self.base_sim.build_directed_from_undirected()
    
    def run_comparison(
        self,
        sender: str,
        receiver: str,
        amount: float = 70000.0,
        p_willing: float = 0.6,
        min_path_nodes: int = 2,
        max_paths_checked: int = 1000,
        include_endpoints_in_share: bool = False,
        verbose: bool = False
    ) -> ComparisonResult:
        """
        Run both simulation modes and compare results.
        
        Args:
            sender: Sender node ID
            receiver: Receiver node ID
            amount: Payment amount in satoshis
            p_willing: Probability of willingness to share info
            min_path_nodes: Minimum nodes per path
            max_paths_checked: Maximum paths to check
            include_endpoints_in_share: Whether to include endpoints in sharing
            verbose: Whether to print detailed output
            
        Returns:
            ComparisonResult with timing and efficiency metrics
        """
        self._setup_base_simulation()
        
        print("=" * 80)
        print("LIGHTNING NETWORK ROUTING COMPARISON")
        print("=" * 80)
        print(f"Sender: {sender}")
        print(f"Receiver: {receiver}")
        print(f"Payment Amount: {amount:,} satoshis")
        print(f"Willingness Probability: {p_willing:.1%}")
        print(f"Min Path Nodes: {min_path_nodes}")
        print(f"Max Paths Checked: {max_paths_checked}")
        print("=" * 80)
        
        # Run simulation WITHOUT info sharing
        print("\n🔄 Running simulation WITHOUT information sharing...")
        sim_no_sharing = SenderViewSim(
            json_file=self.json_file, 
            seed=self.seed, 
            mode=SimulationMode.NO_INFO_SHARING,
            verbose=verbose
        )
        # Copy the pre-built graphs
        sim_no_sharing.G = self.base_sim.G.copy()
        sim_no_sharing.G_dir = self.base_sim.G_dir.copy()
        
        result_no_sharing = sim_no_sharing.run_simulation(
            amount=amount,
            p_willing=p_willing,
            min_path_nodes=min_path_nodes,
            max_paths_checked=max_paths_checked,
            include_endpoints_in_share=include_endpoints_in_share,
            seed_sender_receiver=(sender, receiver)
        )
        
        # Run simulation WITH info sharing
        print("\n🔄 Running simulation WITH information sharing...")
        sim_with_sharing = SenderViewSim(
            json_file=self.json_file, 
            seed=self.seed, 
            mode=SimulationMode.WITH_INFO_SHARING,
            verbose=verbose
        )
        # Copy the pre-built graphs
        sim_with_sharing.G = self.base_sim.G.copy()
        sim_with_sharing.G_dir = self.base_sim.G_dir.copy()
        
        result_with_sharing = sim_with_sharing.run_simulation(
            amount=amount,
            p_willing=p_willing,
            min_path_nodes=min_path_nodes,
            max_paths_checked=max_paths_checked,
            include_endpoints_in_share=include_endpoints_in_share,
            seed_sender_receiver=(sender, receiver)
        )
        
        # Calculate comparison metrics
        time_saved = result_no_sharing.execution_time - result_with_sharing.execution_time
        time_saved_percentage = (time_saved / result_no_sharing.execution_time * 100) if result_no_sharing.execution_time > 0 else 0
        
        paths_saved = result_no_sharing.paths_checked - result_with_sharing.paths_checked
        paths_saved_percentage = (paths_saved / result_no_sharing.paths_checked * 100) if result_no_sharing.paths_checked > 0 else 0
        
        comparison = ComparisonResult(
            no_sharing=result_no_sharing,
            with_sharing=result_with_sharing,
            time_saved=time_saved,
            time_saved_percentage=time_saved_percentage,
            paths_saved=paths_saved,
            paths_saved_percentage=paths_saved_percentage
        )
        
        self._print_comparison_results(comparison)
        return comparison
    
    def _print_comparison_results(self, comparison: ComparisonResult) -> None:
        """Print formatted comparison results."""
        print("\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)
        
        # Success status
        print(f"\n📊 SUCCESS RATE:")
        print(f"  Without Info Sharing: {'✅ SUCCESS' if comparison.no_sharing.success else '❌ FAILED'}")
        print(f"  With Info Sharing:    {'✅ SUCCESS' if comparison.with_sharing.success else '❌ FAILED'}")
        
        # Execution time
        print(f"\n⏱️  EXECUTION TIME:")
        print(f"  Without Info Sharing: {comparison.no_sharing.execution_time:.4f} seconds")
        print(f"  With Info Sharing:    {comparison.with_sharing.execution_time:.4f} seconds")
        print(f"  Time Saved:           {comparison.time_saved:.4f} seconds ({comparison.time_saved_percentage:+.1f}%)")
        
        # Paths checked
        print(f"\n🛤️  PATHS CHECKED:")
        print(f"  Without Info Sharing: {comparison.no_sharing.paths_checked} paths")
        print(f"  With Info Sharing:    {comparison.with_sharing.paths_checked} paths")
        print(f"  Paths Saved:          {comparison.paths_saved} paths ({comparison.paths_saved_percentage:+.1f}%)")
        
        # Efficiency analysis
        print(f"\n📈 EFFICIENCY ANALYSIS:")
        if comparison.time_saved > 0:
            print(f"  ✅ Info sharing SAVED {comparison.time_saved:.4f}s ({comparison.time_saved_percentage:.1f}%)")
        elif comparison.time_saved < 0:
            print(f"  ❌ Info sharing COST {abs(comparison.time_saved):.4f}s ({abs(comparison.time_saved_percentage):.1f}%)")
        else:
            print(f"  ⚖️  No time difference")
            
        if comparison.paths_saved > 0:
            print(f"  ✅ Info sharing checked {comparison.paths_saved} fewer paths ({comparison.paths_saved_percentage:.1f}%)")
        elif comparison.paths_saved < 0:
            print(f"  ❌ Info sharing checked {abs(comparison.paths_saved)} more paths ({abs(comparison.paths_saved_percentage):.1f}%)")
        else:
            print(f"  ⚖️  Same number of paths checked")
        
        # Final verdict
        print(f"\n🎯 VERDICT:")
        if comparison.time_saved > 0 and comparison.paths_saved >= 0:
            print(f"  🏆 Info sharing is MORE EFFICIENT!")
        elif comparison.time_saved < 0 or comparison.paths_saved < 0:
            print(f"  ⚠️  Info sharing is LESS EFFICIENT!")
        else:
            print(f"  🤷 Info sharing has NEUTRAL impact")
        
        print("=" * 80)


# =============================================================================
# Main
# =============================================================================

def get_random_sender_receiver_pairs(
    json_file: str = "LNdata.json", 
    num_pairs: int = 5, 
    seed: Optional[int] = 42
) -> List[Tuple[str, str]]:
    """
    Generate random sender/receiver pairs from the Lightning Network data.
    
    Args:
        json_file: Path to the Lightning Network JSON data
        num_pairs: Number of pairs to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of (sender, receiver) tuples
    """
    # Load data to get available nodes
    base_sim = LightningNetworkBase(json_file=json_file, seed=seed)
    base_sim.load_data()
    base_sim.build_undirected()
    
    nodes = list(base_sim.G.nodes())
    rng = random.Random(seed)
    
    pairs = []
    for _ in range(num_pairs):
        sender, receiver = rng.sample(nodes, 2)
        pairs.append((sender, receiver))
    
    return pairs

def main():
    """Run the comparison between info sharing modes with multiple sender/receiver pairs."""
    seed = 420  # Set to None for true randomness
    
    # Define sender/receiver pairs to test
    # You can modify these pairs as needed
    sender_receiver_pairs = [
        # Add your specific pairs here, or use random generation
        # ("node_id_1", "node_id_2"),
        # ("03a42d6da845fa550db7f79cb8f3cc6ad6f776e7d75111edeef96f3c1e0c152294", "02b9cc0ff32d93fd2f235ab38e4bad0ea393c90aa63fd757f2f0df09110389038f"),
        # ("02e63191b7e52c74cd8cfc666eea44cc6401e6912e2db7ba030a931beb90f17240", "03eab3f5b1955fbebd3019b0e9ce41412884afc2b22ef05017f8e957b93170e41c"),
        ("02dd04e277bf2ee7e8cd8e9766d5615005ac19fa872be8cdfe612a9950ada27b70", "0221ac650cdfd2a7a45102536fa95c3fd9f203de5ff5648f527181082ddd33097a"),
        # ("0354309594599317ac9bea94c845f2212579afbdd6ed7f2bd972f4c9182f00899a", "0261867ef8d1297270157ab3d01fffa4c51ef2435a98186e41e705b0918ed695a2"),
        # ("0362503d3d158444d9ded9a88a6e4308f75d17d96e453c911296cd514a5551f10c", "03e07c085c07e494fd4680b58e7bb51aec5375d935ab0c571562668a33379504d7"),
        # ("03f3524a54dd984f9586df24cc3adb20dce0932e2d7e5eaeb4388b3440319ffff5", "03b3343114f4331076b045d2ffd7e6fea10ef6ec5183a25939833e00131c2f3dfd"),
        # ("02f93201a24036d1c91ec08f656b1d565cdbd0dc1e1f35f44911662a9e03f13cf9", "03facb07b0923165778ea4cd8a59f07dce1000fc5810f5beb82a4a7f852a4b453f"),
        # ("03f3524a54dd984f9586df24cc3adb20dce0932e2d7e5eaeb4388b3440319ffff5", "03b3343114f4331076b045d2ffd7e6fea10ef6ec5183a25939833e00131c2f3dfd"),
    ]
    
    # If no specific pairs provided, generate random ones
    if not sender_receiver_pairs:
        print("No specific pairs provided, generating random pairs...")
        sender_receiver_pairs = get_random_sender_receiver_pairs(
            json_file="LNdata.json", 
            num_pairs=20, 
            seed=seed
        )
    
    print(f"Testing {len(sender_receiver_pairs)} sender/receiver pairs...")
    
    comparison = SimulationComparison(json_file="LNdata.json", seed=seed)
    
    all_results = []
    
    for i, (sender, receiver) in enumerate(sender_receiver_pairs, 1):
        print(f"\n{'='*100}")
        print(f"PAIR {i}/{len(sender_receiver_pairs)}: {sender} -> {receiver}")
        print(f"{'='*100}")
        
        # Run comparison for this pair
        result = comparison.run_comparison(
            sender=sender,
            receiver=receiver,
            amount=5000.0,               # payment amount
            p_willing=0.6,                # willingness to share info
            min_path_nodes=2,             # minimum path length
            max_paths_checked=1000,       # maximum paths to check
            include_endpoints_in_share=False,  # endpoints already known
            verbose=True                 # set to True for detailed output
        )
        
        all_results.append((sender, receiver, result))
    
    # Print summary of all results
    print(f"\n{'='*100}")
    print("SUMMARY OF ALL PAIRS")
    print(f"{'='*100}")
    
    successful_pairs = 0
    total_time_saved = 0.0
    total_paths_saved = 0
    
    for sender, receiver, result in all_results:
        status = "✅" if (result.no_sharing.success and result.with_sharing.success) else "❌"
        print(f"{status} {sender} -> {receiver}: "
              f"Time saved: {result.time_saved:.4f}s ({result.time_saved_percentage:+.1f}%), "
              f"Paths saved: {result.paths_saved} ({result.paths_saved_percentage:+.1f}%)")
        
        if result.no_sharing.success and result.with_sharing.success:
            successful_pairs += 1
            total_time_saved += result.time_saved
            total_paths_saved += result.paths_saved
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Successful pairs: {successful_pairs}/{len(all_results)}")
    if successful_pairs > 0:
        avg_time_saved = total_time_saved / successful_pairs
        avg_paths_saved = total_paths_saved / successful_pairs
        print(f"  Average time saved: {avg_time_saved:.4f}s")
        print(f"  Average paths saved: {avg_paths_saved:.1f}")
    
    return all_results

if __name__ == "__main__":
    main()
