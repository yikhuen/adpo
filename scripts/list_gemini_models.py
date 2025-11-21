#!/usr/bin/env python3
"""Script to list available Gemini models from the Google Generative AI API."""

import os
import sys
from typing import List, Dict, Any

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai package is not installed.")
    print("Install it with: pip install google-generativeai")
    sys.exit(1)


def list_gemini_models(api_key: str = None) -> List[Dict[str, Any]]:
    """List all available Gemini models.
    
    Args:
        api_key: Gemini API key. If not provided, will try to get from GEMINI_API_KEY env var.
    
    Returns:
        List of model dictionaries with model information.
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not set. "
            "Either pass it as an argument or set it as an environment variable."
        )
    
    genai.configure(api_key=api_key)
    
    print("Fetching available Gemini models...")
    print("-" * 80)
    
    try:
        models = genai.list_models()
        return list(models)
    except Exception as e:
        print(f"Error listing models: {e}")
        raise


def print_models(models: List[Dict[str, Any]]) -> None:
    """Print model information in a readable format."""
    if not models:
        print("No models found.")
        return
    
    print(f"\nFound {len(models)} available model(s):\n")
    
    for i, model in enumerate(models, 1):
        name = model.name if hasattr(model, 'name') else getattr(model, 'display_name', 'Unknown')
        # Remove 'models/' prefix if present for cleaner display
        display_name = name.replace('models/', '') if name.startswith('models/') else name
        
        print(f"{i}. {display_name}")
        
        # Print additional attributes if available
        if hasattr(model, 'display_name') and model.display_name != name:
            print(f"   Display Name: {model.display_name}")
        
        if hasattr(model, 'supported_generation_methods'):
            methods = model.supported_generation_methods
            if methods:
                print(f"   Supported Methods: {', '.join(methods)}")
        
        if hasattr(model, 'description'):
            print(f"   Description: {model.description}")
        
        print()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="List available Gemini models from Google Generative AI API"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key (default: use GEMINI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter models by name (case-insensitive substring match)"
    )
    
    args = parser.parse_args()
    
    try:
        models = list_gemini_models(api_key=args.api_key)
        
        # Filter if requested
        if args.filter:
            filter_lower = args.filter.lower()
            models = [
                m for m in models
                if filter_lower in str(m.name).lower() or 
                   (hasattr(m, 'display_name') and filter_lower in str(m.display_name).lower())
            ]
            print(f"Filtered to models containing '{args.filter}':\n")
        
        print_models(models)
        
        # Also print a summary of model names only
        print("\n" + "=" * 80)
        print("Summary - Model names only:")
        print("=" * 80)
        for model in models:
            name = model.name if hasattr(model, 'name') else getattr(model, 'display_name', 'Unknown')
            display_name = name.replace('models/', '') if name.startswith('models/') else name
            print(f"  {display_name}")
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

