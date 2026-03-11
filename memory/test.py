# memory/test.py
import asyncio
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memory import (
    read, write, write_semantic, compress_results, qdrant
)

async def test():
    print("=== Writing test data ===")
    await write_semantic("hardware", "laptop", "HP Omen RTX 4060 32GB RAM")
    await write_semantic("personal", "birthday", "May 9th")
    await write_semantic("project", "current_project", "Building FRIDAY AI assistant")
    await write("what is python", "Python is a high-level programming language")
    await write("open spotify", "Opening Spotify for you")
    print("Data written\n")

    print("=== Testing retrieval ===")
    queries = [
        "what laptop does pavan have",
        "when is pavan birthday",
        "what is pavan building",
        "tell me about python",
        "play some music",
        "what is quantum physics"  # should trigger web fallback
    ]

    for q in queries:
        print(f"Query: '{q}'")
        results = await read(q)
        compressed = compress_results(results)
        source = results[0]["source"] if results else "none"
        print(f"  Source: {source}")
        print(f"  Result: {compressed[:100] if compressed else 'Nothing found'}")
        print()

    qdrant.close()

asyncio.run(test())