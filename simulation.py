import numpy as np
import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CORE SIMULATION CLASSES
# ============================================================================

@dataclass
class CascadeResult:
    """Results from a single cascade simulation."""
    cascade_size: int
    cascade_depth: int
    infection_timeline: List[int]
    ro: float
    extinct: bool

class NetworkGenerator:
    """Generate different network topologies"""
    @staticmethod
    def generate_network(n=1000, network_type="BA", parameters=None):
        if parameters is None:
            parameters = {}
        if network_type == "ER":
            p = parameters.get("p", 0.005)
            G = nx.erdos_renyi_graph(n, p)
        elif network_type == "BA":
            m = parameters.get("m", 3)
            G = nx.barabasi_albert_graph(n, m)
        elif network_type == "WS":
            k = parameters.get("k", 6)
            p = parameters.get("p", 0.1)
            G = nx.watts_strogatz_graph(n, k, p)
        else:
            G = nx.barabasi_albert_graph(n, 3)
            
        # Ensure connected
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        return G

class PropagationModel:
    """Simple propagation model with credibility factor"""
    def __init__(self, p_base=0.1, content_type="fake"):
        self.p_base = p_base
        self.content_type = content_type
        # Fake news spreads more easily (credibility factor 0.3 vs 1.0)
        self.credibility_factor = 0.3 if content_type == "fake" else 1.0

    def sharing_probability(self, n_exposures):
        if n_exposures == 0:
            return 0.0
        prob_no_share = (1 - self.p_base) ** n_exposures
        prob_no_share *= self.credibility_factor
        return min(1 - prob_no_share, 1.0)

class CascadeSimulator:
    """Simulate cascades with optional fact-checking intervention"""
    def __init__(self, network, propagation_model, intervention=None):
        self.network = network
        self.model = propagation_model
        self.intervention = intervention or {}
        self.states = {node: "susceptible" for node in network.nodes()}
        self.exposures = defaultdict(int)
        self.infection_times = {}
        self.fact_checkers = set()
        
        # Apply fact-checking if specified
        if self.intervention.get("type") in ["random_factcheck", "targeted_factcheck"]:
            self._apply_factcheck_intervention()

    def _apply_factcheck_intervention(self):
        """Apply fact-checking intervention"""
        intensity = self.intervention.get("intensity", 0.0)
        n_factcheckers = max(1, int(len(self.network.nodes()) * intensity))
        
        if self.intervention["type"] == "random_factcheck":
            # Random selection
            all_nodes = list(self.network.nodes())
            self.fact_checkers = set(np.random.choice(all_nodes, min(n_factcheckers, len(all_nodes)), replace=False))
        elif self.intervention["type"] == "targeted_factcheck":
            # High-degree nodes (influencers)
            degrees = dict(self.network.degree())
            top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
            self.fact_checkers = set(top_nodes[:min(n_factcheckers, len(top_nodes))])

    def simulate(self, seed_node=None, max_rounds=50):
        """Run cascade simulation"""
        if seed_node is None:
            seed_node = np.random.choice(list(self.network.nodes()))
        self.states[seed_node] = "infected"
        self.infection_times[seed_node] = 0
        infection_timeline = [1]
        current_round = 0
        
        while current_round < max_rounds:
            current_round += 1
            new_infections = []
            
            # Get currently infected nodes
            infected_nodes = [n for n, s in self.states.items() if s == "infected"]
            if not infected_nodes:
                break
                
            # Spread to neighbors
            for infected in infected_nodes:
                for neighbor in self.network.neighbors(infected):
                    if self.states[neighbor] == "susceptible":
                        self.exposures[neighbor] += 1
                        self.states[neighbor] = "exposed"
                        
            # Exposed nodes decide to share
            for node in [n for n, s in self.states.items() if s == "exposed"]:
                # Fact-checkers don't share fake news.
                if node in self.fact_checkers and self.model.content_type == "fake":
                    continue
                n_exp = self.exposures[node]
                p_share = self.model.sharing_probability(n_exp)
                if np.random.random() < p_share:
                    self.states[node] = "infected"
                    self.infection_times[node] = current_round
                    new_infections.append(node)
                    
            # Move infected to recovered
            for infected in infected_nodes:
                self.states[infected] = "recovered"
            infection_timeline.append(len(new_infections))
            
            if len(new_infections) == 0:
                break
                
        cascade_size = sum(1 for s in self.states.values() if s in ["infected", "recovered"])
        # Calculate RO (simplified)
        ro = np.mean([len(list(self.network.neighbors(n))) for n in self.infection_times.keys()])
        
        return CascadeResult(
            cascade_size=cascade_size,
            cascade_depth=current_round,
            infection_timeline=infection_timeline,
            ro=ro,
            extinct=cascade_size < 10
        )

# ============================================================================
# 3/10/30 RULE EXPERIMENT
# ============================================================================

class EfficacyExperiment:
    """Run the 3/10/30 rule experiment"""
    def __init__(self, n_nodes=300, n_simulations=150, network_type="BA"):
        self.n_nodes = n_nodes
        self.n_simulations = n_simulations
        self.network_type = network_type
        
        # Generate base network (same for all simulations)
        print(f"Generating {network_type} network with {n_nodes} nodes...")
        self.network = NetworkGenerator.generate_network(
            n=n_nodes,
            network_type=network_type,
            parameters={"m": 3} if network_type == "BA" else {}
        )
        print(f"Network ready: {self.network.number_of_nodes()} nodes, {self.network.number_of_edges()} edges")

    def run_condition(self, intervention_type=None, intensity=0.0) -> List[CascadeResult]:
        """Run multiple simulations for a specific condition"""
        results = []
        for i in range(self.n_simulations):
            if (i+1) % 50 == 0:
                print(f" Progress: {i+1}/{self.n_simulations}")
            
            # Reset random seed variation for diversity
            np.random.seed(i + 42)
            
            # Create intervention config
            intervention = None
            if intervention_type:
                intervention = {
                    "type": intervention_type,
                    "intensity": intensity
                }
                
            # Create fresh model and simulator for each run
            model = PropagationModel(p_base=0.1, content_type="fake")
            simulator = CascadeSimulator(self.network, model, intervention)
            result = simulator.simulate()
            results.append(result)
            
        return results

    def run_experiment(self) -> Dict:
        """Run all conditions and compare"""
        print("\n" + "="*60)
        print("RUNNING 3/10/30 RULE EXPERIMENT")
        print("="*60)
        
        # Baseline (no intervention)
        print("\n[1/7] Baseline (no intervention)...")
        baseline_results = self.run_condition(intervention_type=None)
        baseline_size = np.mean([r.cascade_size for r in baseline_results])
        print(f" Baseline cascade size: {baseline_size:.1f}")
        
        # Random fact-checking
        random_intensities = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
        random_results = {}
        print("\n[2/7] Random fact-checking...")
        for intensity in random_intensities:
            print(f" Intensity: {intensity*100:.0f}%")
            results = self.run_condition("random_factcheck", intensity)
            mean_size = np.mean([r.cascade_size for r in results])
            reduction = (baseline_size - mean_size) / baseline_size * 100
            random_results[intensity] = {
                'results': results,
                'mean_size': mean_size,
                'reduction': reduction
            }
            print(f" Mean size: {mean_size:.1f} (reduction: {reduction:.1f}%)")
            
        # Targeted fact-checking
        targeted_intensities = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
        targeted_results = {}
        print("\n[3/7] Targeted fact-checking (high-degree nodes)...")
        for intensity in targeted_intensities:
            print(f" Intensity: {intensity*100:.1f}%")
            results = self.run_condition("targeted_factcheck", intensity)
            mean_size = np.mean([r.cascade_size for r in results])
            reduction = (baseline_size - mean_size) / baseline_size * 100
            targeted_results[intensity] = {
                'results': results,
                'mean_size': mean_size,
                'reduction': reduction
            }
            print(f" Mean size: {mean_size:.1f} (reduction: {reduction:.1f}%)")
            
        return {
            'baseline_size': baseline_size,
            'baseline_results': baseline_results,
            'random': random_results,
            'targeted': targeted_results
        }

    def analyze_and_print_results(self, results: Dict):
        """Analyze and print results"""
        baseline = results['baseline_size']
        
        # Get targeted 10% reduction
        targeted_10_reduction = results['targeted'][0.10]['reduction']
        
        # Find random intensity that matches
        random_intensities = sorted(results['random'].keys())
        random_reductions = [results['random'][i]['reduction'] for i in random_intensities]
        
        # Interpolation
        equivalent_random = np.interp(targeted_10_reduction, random_reductions, random_intensities)
        efficiency_gain = equivalent_random / 0.10
        
        # Get random 30% reduction
        random_30_reduction = np.interp(0.30, random_intensities, random_reductions)
        
        # Print summary
        print("\n" + "="*70)
        print(" 3/10/30 RULE EXPERIMENT RESULTS")
        print("="*70)
        print(f"\n KEY FINDING:")
        print(f" • Targeted fact-checking at 10% reduces cascades by ~{targeted_10_reduction:.1f}%")
        print(f" • Random fact-checking needs {equivalent_random*100:.0f}% to achieve same")
        print(f" → EFFICIENCY GAIN: {efficiency_gain:.1f}x")
        print(f"\n COMPARISON:")
        print(f" • Random 30% reduces by ~{random_30_reduction:.1f}%")
        print(f" • Targeted 10% reduces by ~{targeted_10_reduction:.1f}%")
        print(f"\n PRACTICAL IMPLICATION:")
        print(f" Social media platforms can achieve the same fake news suppression")
        print(f" by fact-checking the top 10% most connected users (influencers)")
        print(f" instead of {equivalent_random*100:.0f}% random users.")
        print(f" This saves {(1 - 0.1/equivalent_random)*100:.0f}% of fact-checking resources.")
        print("\n" + "="*70)
        
        # Show detailed table
        print("\n DETAILED RESULTS TABLE:")
        print("-"*65)
        print(f"{'Type': <12} {'Intensity': <12} {'Mean Size': <15} {'Reduction %': <12}")
        print("-"*65)
        print(f"{'Baseline': <12} {'0%':<12} {baseline:<15.1f} {'0.0':<12}")
        for intensity, info in results['random'].items():
            print(f"{'Random': <12} {int(intensity*100):<12}% {info['mean_size']:<15.1f} {info['reduction']:<12.1f}")
        for intensity, info in results['targeted'].items():
            print(f"{'Targeted': <12} {int(intensity*100):<12}% {info['mean_size']:<15.1f} {info['reduction']:<12.1f}")
        print("-"*65)
        
        # Save to CSV
        data_rows = []
        for intensity, info in results['random'].items():
            data_rows.append({
                'type': 'random',
                'intensity_percent': intensity * 100,
                'mean_cascade_size': info['mean_size'],
                'reduction_percent': info['reduction']
            })
        for intensity, info in results['targeted'].items():
            data_rows.append({
                'type': 'targeted',
                'intensity_percent': intensity * 100,
                'mean_cascade_size': info['mean_size'],
                'reduction_percent': info['reduction']
            })
            
        df = pd.DataFrame(data_rows)
        df.to_csv("3_10_30_rule_results.csv", index=False)
        print("\n Results saved to: 3_10_30_rule_results.csv")
        
        return efficiency_gain

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the complete experiment"""
    print("\n" + "="*70)
    print(" 3/10/30 RULE EXPERIMENT")
    print("Quantifying targeted vs random fact-checking efficiency")
    print("="*70)
    
    # Parameters
    N_NODES = 300      # Network size
    N_SIMS = 150       # Simulations per condition
    
    print(f"\n Configuration:")
    print(f" Network size: {N_NODES} nodes")
    print(f" Simulations per condition: {N_SIMS}")
    print(f" Network type: Barabasi-Albert (scale-free)")
    print(f" Propagation: p_base = 0.1, fake news credibility = 0.3")
    
    # Run experiment
    exp = EfficacyExperiment(n_nodes=N_NODES, n_simulations=N_SIMS, network_type="BA")
    results = exp.run_experiment()
    
    # Analyze and print
    efficiency = exp.analyze_and_print_results(results)
    print(f"\n Experiment complete! Efficiency gain: {efficiency:.1f}x")
    return results

# ============================================================================
# QUICK RUN WITH PRE-COMPUTED RESULTS (based on your actual run)
# ============================================================================

def quick_results():
    """Use your actual simulation results without re-running"""
    print("\n" + "="*70)
    print(" 3/10/30 RULE - YOUR ACTUAL RESULTS")
    print("="*70)
    
    # Your actual results from the simulation
    baseline_size = 293.7
    random_results = {
        0.05: {'reduction': 3.7, 'mean_size': 282.7},
        0.10: {'reduction': 10.0, 'mean_size': 264.5},
        0.15: {'reduction': 15.1, 'mean_size': 249.5},
        0.20: {'reduction': 20.8, 'mean_size': 232.8},
        0.25: {'reduction': 28.0, 'mean_size': 211.5},
        0.30: {'reduction': 33.6, 'mean_size': 194.9},
        0.35: {'reduction': 41.0, 'mean_size': 173.4},
    }
    targeted_results = {
        0.01: {'reduction': 1.6, 'mean_size': 289.0},
        0.02: {'reduction': 3.3, 'mean_size': 284.0},
        0.03: {'reduction': 6.2, 'mean_size': 275.4},
        0.05: {'reduction': 12.9, 'mean_size': 255.8},
        0.07: {'reduction': 15.7, 'mean_size': 247.6},
        0.10: {'reduction': 22.6, 'mean_size': 227.2},
        0.15: {'reduction': 37.2, 'mean_size': 184.4},
        0.20: {'reduction': 72.9, 'mean_size': 79.5},
    }
    
    # Calculate key metrics
    targeted_10_reduction = targeted_results[0.10]['reduction']
    
    # Find equivalent random intensity
    random_intensities = sorted(random_results.keys())
    random_reductions = [random_results[i]['reduction'] for i in random_intensities]
    equivalent_random = np.interp(targeted_10_reduction, random_reductions, random_intensities)
    efficiency_gain = equivalent_random / 0.10
    
    # Random 30% reduction
    random_30_reduction = np.interp(0.30, random_intensities, random_reductions)
    
    print(f"\n KEY FINDING:")
    print(f" • Targeted fact-checking at 10% reduces cascades by ~{targeted_10_reduction:.1f}%")
    print(f" • Random fact-checking needs {equivalent_random*100:.0f}% to achieve similar results")
    print(f" → EFFICIENCY GAIN: {efficiency_gain:.1f}x")
    print(f"\n COMPARISON:")
    print(f" • Random 30% reduces by ~{random_30_reduction:.1f}%")
    print(f" • Targeted 10% reduces by ~{targeted_10_reduction:.1f}%")
    print(f"\n PRACTICAL IMPLICATION:")
    print(f" Fact-checking top 10% most connected users achieves")
    print(f" the same suppression as fact-checking {equivalent_random*100:.0f}% random users.")
    print(f" Resource savings: {(1 - 0.1/equivalent_random)*100:.0f}%")
    print("\n" + "="*70)
    
    # Print detailed table
    print("\n DETAILED RESULTS TABLE:")
    print("-"*65)
    print(f"{'Type': <12} {'Intensity': <12} {'Mean Size': <15} {'Reduction %': <12}")
    print("-"*65)
    print(f"{'Baseline': <12} {'0%': <12} {baseline_size:<15.1f} {'0.0':<12}")
    for intensity, info in random_results.items():
        print(f"{'Random': <12} {int(intensity*100):<12}% {info['mean_size']:<15.1f} {info['reduction']:<12.1f}")
    for intensity, info in targeted_results.items():
        print(f"{'Targeted': <12} {int(intensity*100):<12}% {info['mean_size']:<15.1f} {info['reduction']:<12.1f}")
    print("-"*65)
    
    # Save to CSV
    data_rows = []
    for intensity, info in random_results.items():
        data_rows.append({
            'type': 'random',
            'intensity_percent': intensity * 100,
            'mean_cascade_size': info['mean_size'],
            'reduction_percent': info['reduction']
        })
    for intensity, info in targeted_results.items():
        data_rows.append({
            'type': 'targeted',
            'intensity_percent': intensity * 100,
            'mean_cascade_size': info['mean_size'],
            'reduction_percent': info['reduction']
        })
        
    df = pd.DataFrame(data_rows)
    df.to_csv("3_10_30_rule_results.csv", index=False)
    print("\n Results saved to: 3_10_30_rule_results.csv")
    
    return efficiency_gain

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Select mode:")
    print("1. Run full simulation (takes 2-5 minutes)")
    print("2. Use pre-computed results (instant, from your run)")
    print("="*70)
    
    choice = input("\nEnter 1 or 2 (default: 2): ").strip()
    
    if choice == "1":
        results = main()
    else:
        efficiency = quick_results()
