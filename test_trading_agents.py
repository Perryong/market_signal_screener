"""
Quick test script to verify TradingAgents integration is working.
"""

import os
from dotenv import load_dotenv
from config import TRADING_AGENTS_CONFIG
from trading_agents_integration import initialize_trading_agents

def test_trading_agents():
    """Test if TradingAgents can be initialized."""
    print("="*60)
    print("Testing TradingAgents Integration")
    print("="*60)
    
    # Check config
    print(f"\n1. Config check:")
    print(f"   Enabled: {TRADING_AGENTS_CONFIG.get('enabled', False)}")
    print(f"   Analysts: {TRADING_AGENTS_CONFIG.get('selected_analysts', [])}")
    
    # Check environment
    print(f"\n2. Environment check:")
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print(f"   ✓ OPENAI_API_KEY found (length: {len(openai_key)})")
    else:
        print(f"   ✗ OPENAI_API_KEY not found in .env file")
        print(f"   Note: TradingAgents requires OPENAI_API_KEY to work")
    
    # Try to import
    print(f"\n3. Import check:")
    try:
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        print(f"   ✓ TradingAgentsGraph imported successfully")
    except ImportError as e:
        print(f"   ✗ Failed to import TradingAgentsGraph: {e}")
        return False
    
    # Try to initialize
    print(f"\n4. Initialization check:")
    if not TRADING_AGENTS_CONFIG.get("enabled", True):
        print("   TradingAgents is disabled in config")
        return False
    
    try:
        ta_config = {
            "deep_think_llm": TRADING_AGENTS_CONFIG.get("deep_think_llm", "gpt-4o-mini"),
            "quick_think_llm": TRADING_AGENTS_CONFIG.get("quick_think_llm", "gpt-4o-mini"),
            "max_debate_rounds": TRADING_AGENTS_CONFIG.get("max_debate_rounds", 1),
            "data_vendors": TRADING_AGENTS_CONFIG.get("data_vendors", {}),
        }
        
        ta_graph = initialize_trading_agents(
            config=ta_config,
            debug=False,
            selected_analysts=TRADING_AGENTS_CONFIG.get("selected_analysts", [])
        )
        
        if ta_graph:
            print(f"   ✓ TradingAgentsGraph initialized successfully")
            print(f"   ✓ Ready to analyze tickers")
            return True
        else:
            print(f"   ✗ TradingAgentsGraph initialization returned None")
            return False
            
    except Exception as e:
        print(f"   ✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_trading_agents()
    print("\n" + "="*60)
    if success:
        print("✓ TradingAgents integration test PASSED")
    else:
        print("✗ TradingAgents integration test FAILED")
        print("\nTroubleshooting:")
        print("1. Check that all dependencies are installed: pip install -r requirements.txt")
        print("2. Verify OPENAI_API_KEY is set in .env file")
        print("3. Check that TradingAgents is enabled in config.py")
    print("="*60)
