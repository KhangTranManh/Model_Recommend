#!/usr/bin/env python3
"""
Main launcher for Fashion Recommendation System
This is the primary entry point for running the application.
"""

import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Fashion Recommendation System")
    parser.add_argument("--mode", "-m", choices=["gui", "cli", "setup"], default="gui",
                       help="Run mode: gui (default), cli, or setup")
    parser.add_argument("--query", "-q", type=str, help="Text query (CLI mode)")
    parser.add_argument("--image", "-i", type=str, help="Image path (CLI mode)")
    parser.add_argument("--k", type=int, default=6, help="Number of results (CLI mode)")
    parser.add_argument("--show", action="store_true", help="Show images (CLI mode)")
    
    args = parser.parse_args()
    
    # Add project root to path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    print("🎨 Fashion Recommendation System")
    print("=" * 40)
    
    if args.mode == "setup":
        print("🔧 Running setup verification...")
        os.system(f"python {os.path.join('scripts', 'setup_environment.py')}")
        
    elif args.mode == "cli":
        print("💻 Starting CLI mode...")
        cli_args = []
        if args.query:
            cli_args.extend(["--query", args.query])
        if args.image:
            cli_args.extend(["--image", args.image])
        if args.k != 6:
            cli_args.extend(["--k", str(args.k)])
        if args.show:
            cli_args.append("--show")
            
        cmd = f"python {os.path.join('scripts', 'run_cli.py')} {' '.join(cli_args)}"
        os.system(cmd)
        
    else:  # gui mode (default)
        print("🖥️ Starting GUI mode...")
        os.system(f"python {os.path.join('scripts', 'run_gui.py')}")

if __name__ == "__main__":
    main()