"""Patched entry point that uses custom_topology to generate the graph.

Instead of reading hardcoded edges from config.yaml, this script:
1. Calls generate_topology(node_num) from custom_topology.py
2. Writes the resulting edges into config.yaml's graph field
3. Proceeds with normal MacNet Graph construction and execution
"""
import argparse
import os
import sys
import yaml

# Ensure the workspace root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from custom_topology import generate_topology
from graph import Graph


def load_config():
    """Load configuration from YAML file."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--task', type=str, default="Develop a basic Gomoku game.",
                        help="Prompt of software")
    parser.add_argument('--name', type=str, default="Gomoku")
    parser.add_argument('--type', type=str, default="None")
    parser.add_argument('--node_num', type=int, default=4,
                        help="Number of agent nodes in the topology")
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    # Generate topology from custom_topology module
    node_num = int(os.environ.get("NODE_NUM", args.node_num))
    edges = generate_topology(node_num)

    # Convert edges to config.yaml format: ['0->1', '0->2', ...]
    graph_strings = [f"{src}->{tgt}" for src, tgt in edges]

    # Load and update config
    config = load_config()
    config["graph"] = graph_strings

    # Write updated config back
    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Reload config (to ensure yaml formatting is consistent)
    config = load_config()

    # Graph construction and agents deployment
    graph = Graph(config)
    graph.build_graph(args.type)
    graph.agent_deployment(args.type)
    graph.execute(args.task, args.name)

    with open(graph.directory + "/config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    print("MacNet completes!")


if __name__ == "__main__":
    main()
