# Quantifying Fact-Checking Efficiency: The 3/10/30 Rule Experiment

This repository contains a Python simulation that models the propagation of fake news cascades across a network. It specifically quantifies the efficiency gain of **targeted fact-checking** (focusing on high-degree nodes or "influencers") versus **random fact-checking**.

## 📌 The Core Finding
Social media platforms can achieve the same level of fake news suppression by fact-checking the **top 10% most connected users** as they would by randomly fact-checking roughly **30% of all users**.

## 🔬 How the Simulation Works
The experiment uses a custom propagation model built on top of `networkx`[cite: 3]:
* **Network Topology:** Generates a Barabási-Albert (scale-free) network to mimic real-world social media follower distributions[cite: 3].
* **Propagation Model:** Fake news is assigned a higher "credibility/spread factor" (0.3) compared to standard content (1.0), making it highly contagious[cite: 3].
* **Interventions:** 
  * *Baseline:* No fact-checking intervention[cite: 3].
  * *Random Fact-Checking:* A randomized percentage of nodes are selected as fact-checkers who will not share fake news[cite: 3].
  * *Targeted Fact-Checking:* The highest-degree nodes (hubs/influencers) are selected as fact-checkers[cite: 3].

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.x installed. Install the required dependencies using pip:
```pip install -r requirements.txt
