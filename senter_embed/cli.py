#!/usr/bin/env python3
"""
Senter-Embed CLI Interface

Command-line interface for the Senter-Embed multimodal embedding system.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from .core import SenterEmbedder
from .database import MultimodalEmbeddingDatabase


def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(description="Senter-Embed: Multimodal Embedding System")

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Generate embeddings')
    embed_parser.add_argument('--text', type=str, help='Text to embed')
    embed_parser.add_argument('--image', type=str, help='Image path to embed')
    embed_parser.add_argument('--audio', type=str, help='Audio path to embed')
    embed_parser.add_argument('--video', type=str, help='Video path to embed')
    embed_parser.add_argument('--output', type=str, help='Output file for embeddings')

    # Database commands
    db_parser = subparsers.add_parser('db', help='Database operations')
    db_subparsers = db_parser.add_subparsers(dest='db_command')

    # Add to database
    add_parser = db_subparsers.add_parser('add', help='Add content to database')
    add_parser.add_argument('--text', type=str, help='Text content')
    add_parser.add_argument('--image', type=str, help='Image path')
    add_parser.add_argument('--audio', type=str, help='Audio path')
    add_parser.add_argument('--video', type=str, help='Video path')
    add_parser.add_argument('--db', type=str, required=True, help='Database file path')

    # Search database
    search_parser = db_subparsers.add_parser('search', help='Search database')
    search_parser.add_argument('--text', type=str, help='Text query')
    search_parser.add_argument('--image', type=str, help='Image query path')
    search_parser.add_argument('--audio', type=str, help='Audio query path')
    search_parser.add_argument('--video', type=str, help='Video query path')
    search_parser.add_argument('--db', type=str, required=True, help='Database file path')
    search_parser.add_argument('--top-k', type=int, default=5, help='Number of results')

    # Database info
    info_parser = db_subparsers.add_parser('info', help='Database information')
    info_parser.add_argument('--db', type=str, required=True, help='Database file path')

    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run embedding demo')

    return parser


def run_embed_command(args):
    """Handle embed command"""
    try:
        embedder = SenterEmbedder()

        content = {}
        if args.text:
            content['text'] = args.text
        if args.image:
            content['image'] = args.image
        if args.audio:
            content['audio'] = args.audio
        if args.video:
            content['video'] = args.video

        if not content:
            print("❌ No content provided. Use --text, --image, --audio, or --video")
            return

        embeddings = embedder.embed_multimodal_content(content)

        print("🎯 Generated embeddings:")
        for modality, embedding in embeddings.items():
            print(f"  {modality}: shape {embedding.shape}, norm {torch.norm(embedding):.4f}")

        if args.output:
            # Save embeddings
            result = {
                'content': content,
                'embeddings': {k: v.tolist() for k, v in embeddings.items()}
            }

            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"💾 Embeddings saved to {args.output}")

    except Exception as e:
        print(f"❌ Embedding failed: {e}")


def run_db_command(args):
    """Handle database commands"""
    try:
        embedder = SenterEmbedder()

        if args.db_command == 'add':
            # Add content to database
            db = MultimodalEmbeddingDatabase(embedder)

            content = {}
            if args.text:
                content['text'] = args.text
            if args.image:
                content['image'] = args.image
            if args.audio:
                content['audio'] = args.audio
            if args.video:
                content['video'] = args.video

            if not content:
                print("❌ No content provided to add")
                return

            db.add_content(content)
            db.save_database(args.db)
            print(f"✅ Content added to database: {args.db}")

        elif args.db_command == 'search':
            # Search database
            db = MultimodalEmbeddingDatabase(embedder)
            db.load_database(args.db)

            query = {}
            if args.text:
                query['text'] = args.text
            if args.image:
                query['image'] = args.image
            if args.audio:
                query['audio'] = args.audio
            if args.video:
                query['video'] = args.video

            if not query:
                print("❌ No query provided")
                return

            results = db.search_similar(query, top_k=args.top_k)

            print(f"🔍 Search results (top {args.top_k}):")
            for result in results:
                print(f"  Rank {result['rank']}: {result['similarity']:.3f}")
                print(f"    Modality: {result['modality']}")
                print(f"    Content: {result['content'][:100]}...")
                print()

        elif args.db_command == 'info':
            # Database info
            if not Path(args.db).exists():
                print(f"❌ Database not found: {args.db}")
                return

            db = MultimodalEmbeddingDatabase(embedder)
            db.load_database(args.db)
            stats = db.get_stats()

            print(f"📊 Database Information: {args.db}")
            print(f"  Total embeddings: {stats['total_embeddings']}")
            print(f"  Modalities: {stats['modalities']}")
            if stats['embedding_shape']:
                print(f"  Embedding shape: {stats['embedding_shape']}")

    except Exception as e:
        print(f"❌ Database operation failed: {e}")


def run_demo():
    """Run the embedding demo"""
    from ..simple_embedding_demo import demo_multimodal_embeddings

    print("🎭 Running Senter-Embed Demo")
    demo_multimodal_embeddings()


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == 'embed':
        run_embed_command(args)
    elif args.command == 'db':
        if not hasattr(args, 'db_command') or not args.db_command:
            parser.parse_args(['db', '--help'])
        else:
            run_db_command(args)
    elif args.command == 'demo':
        run_demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
