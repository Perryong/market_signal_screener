"""
Integration module for TradingAgents LLM-based multi-agent trading framework.

This module provides a clean interface between the main technical analysis
application and the TradingAgents framework.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from data_types.signal import Signal


def initialize_trading_agents(
    config: Optional[Dict[str, Any]] = None,
    debug: bool = False,
    selected_analysts: list = None
) -> Optional[TradingAgentsGraph]:
    """
    Initialize TradingAgentsGraph with proper configuration.
    
    Args:
        config: Optional configuration dictionary. If None, uses defaults.
        debug: Whether to run in debug mode
        selected_analysts: List of analyst types to include
        
    Returns:
        Initialized TradingAgentsGraph instance, or None if initialization fails
    """
    try:
        # Start with default config
        ta_config = DEFAULT_CONFIG.copy()
        
        # Merge with provided config
        if config:
            ta_config.update(config)
        
        # Set default analysts if not provided
        if selected_analysts is None:
            selected_analysts = ["market", "social", "news", "fundamentals"]
        
        # Initialize TradingAgentsGraph
        ta_graph = TradingAgentsGraph(
            selected_analysts=selected_analysts,
            debug=debug,
            config=ta_config
        )
        
        print(f"TradingAgents initialized with analysts: {selected_analysts}")
        return ta_graph
        
    except ImportError as e:
        print(f"Error: Failed to import TradingAgents modules: {e}")
        print("Make sure all TradingAgents dependencies are installed: pip install -r requirements.txt")
        import traceback
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"Warning: Failed to initialize TradingAgents: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_with_trading_agents(
    ticker: str,
    analysis_date: str,
    ta_graph: TradingAgentsGraph,
    save_full_trace: bool = True
) -> Tuple[Dict[str, Any], str]:
    """
    Run TradingAgents analysis for a ticker on a specific date.
    
    Args:
        ticker: Stock ticker symbol
        analysis_date: Date string in "YYYY-MM-DD" format
        ta_graph: Initialized TradingAgentsGraph instance
        save_full_trace: Whether to save full agent conversation trace to JSON
        
    Returns:
        Tuple of (final_state, decision) where:
        - final_state: Complete state dictionary from the graph
        - decision: Extracted decision string (BUY/SELL/HOLD)
        
    Raises:
        Exception: If analysis fails
    """
    if ta_graph is None:
        raise ValueError("TradingAgentsGraph is not initialized")
    
    # Capture full trace for JSON export
    trace_data = []
    original_debug = ta_graph.debug
    
    # Temporarily enable debug mode to capture trace
    if save_full_trace:
        ta_graph.debug = True
    
    try:
        # Use the existing propagate method which handles everything
        # But we'll intercept the stream to capture trace
        from tradingagents.graph.propagation import Propagator
        propagator = Propagator()
        init_agent_state = propagator.create_initial_state(ticker, analysis_date)
        args = propagator.get_graph_args()
        
        # Collect trace by streaming
        trace = []
        for chunk in ta_graph.graph.stream(init_agent_state, **args):
            trace.append(chunk)
            
            # Extract message information for JSON
            if "messages" in chunk and len(chunk["messages"]) > 0:
                for msg in chunk["messages"]:
                    msg_data = _extract_message_data(msg)
                    if msg_data:
                        trace_data.append(msg_data)
        
        final_state = trace[-1] if trace else None
        
        # If we didn't get final state from trace, use invoke
        if final_state is None:
            final_state = ta_graph.graph.invoke(init_agent_state, **args)
        
        # Restore original debug setting
        ta_graph.debug = original_debug
        
        # Update ticker and state
        ta_graph.ticker = ticker
        ta_graph.curr_state = final_state
        
        # Log state (existing method)
        ta_graph._log_state(analysis_date, final_state)
        
        # Save full trace to JSON
        if save_full_trace and trace_data:
            _save_full_trace_json(ticker, analysis_date, trace_data, final_state)
        
        # Return decision and processed signal
        decision = ta_graph.process_signal(final_state["final_trade_decision"])
        
        return final_state, decision
        
    except Exception as e:
        # Restore original debug setting on error
        ta_graph.debug = original_debug
        raise


def _extract_message_data(msg) -> Optional[Dict[str, Any]]:
    """
    Extract message data from LangChain message object for JSON serialization.
    
    Args:
        msg: LangChain message object
        
    Returns:
        Dictionary with message data, or None if not extractable
    """
    try:
        msg_type = type(msg).__name__
        data = {
            "type": msg_type,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Extract content
        if hasattr(msg, 'content'):
            content = msg.content
            if isinstance(content, str):
                data["content"] = content
            elif isinstance(content, list):
                # Handle list content (may contain tool calls)
                data["content"] = []
                tool_calls_list = []
                for item in content:
                    if hasattr(item, 'type'):
                        item_type = getattr(item, 'type', '')
                        if item_type == "tool_use" or item_type == "tool_call":
                            tool_calls_list.append({
                                "id": getattr(item, 'id', ''),
                                "name": getattr(item, 'name', ''),
                                "input": getattr(item, 'input', {}),
                                "args": getattr(item, 'args', getattr(item, 'input', {}))
                            })
                        else:
                            data["content"].append(str(item))
                    else:
                        data["content"].append(str(item))
                if tool_calls_list:
                    data["tool_calls"] = tool_calls_list
            else:
                data["content"] = str(content)
        
        # Extract tool calls if present (LangChain format)
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            data["tool_calls"] = []
            for tool_call in msg.tool_calls:
                tool_data = {
                    "name": getattr(tool_call, 'name', ''),
                    "args": getattr(tool_call, 'args', {}),
                    "id": getattr(tool_call, 'id', ''),
                }
                # Try to get more details
                if hasattr(tool_call, '__dict__'):
                    tool_data.update({k: v for k, v in tool_call.__dict__.items() 
                                    if k not in ['name', 'args', 'id'] and not k.startswith('_')})
                data["tool_calls"].append(tool_data)
        
        # Extract tool results if present (ToolMessage)
        if hasattr(msg, 'tool_call_id'):
            data["tool_call_id"] = msg.tool_call_id
            data["name"] = getattr(msg, 'name', '')
        
        # Extract additional attributes
        if hasattr(msg, 'additional_kwargs'):
            additional = msg.additional_kwargs
            if additional:
                data["additional_kwargs"] = additional
        
        # Extract response metadata if present
        if hasattr(msg, 'response_metadata'):
            data["response_metadata"] = msg.response_metadata
        
        # If no content but has string representation, use that
        if "content" not in data or not data["content"]:
            str_repr = str(msg)
            if str_repr and str_repr != f"<{msg_type} object>":
                data["content"] = str_repr[:1000]  # Limit length
        
        return data if data.get("content") or data.get("tool_calls") else None
        
    except Exception as e:
        # If extraction fails, return basic info
        return {
            "type": type(msg).__name__,
            "content": str(msg)[:500],
            "error": f"Failed to extract full message: {str(e)}"
        }


def _save_full_trace_json(
    ticker: str,
    analysis_date: str,
    trace_data: List[Dict[str, Any]],
    final_state: Dict[str, Any]
):
    """
    Save full agent conversation trace to JSON file.
    
    Args:
        ticker: Stock ticker symbol
        analysis_date: Analysis date string
        trace_data: List of message dictionaries
        final_state: Final state from the graph
    """
    try:
        # Create directory
        directory = Path(f"eval_results/{ticker}/TradingAgentsStrategy_logs/")
        directory.mkdir(parents=True, exist_ok=True)
        
        # Prepare comprehensive JSON output
        output = {
            "metadata": {
                "ticker": ticker,
                "analysis_date": analysis_date,
                "timestamp": datetime.now().isoformat(),
                "total_messages": len(trace_data),
                "description": "Full agent conversation trace including all messages, tool calls, and decisions"
            },
            "conversation_trace": trace_data,
            "final_state": {
                "company_of_interest": final_state.get("company_of_interest", ""),
                "trade_date": final_state.get("trade_date", ""),
                "market_report": final_state.get("market_report", ""),
                "sentiment_report": final_state.get("sentiment_report", ""),
                "news_report": final_state.get("news_report", ""),
                "fundamentals_report": final_state.get("fundamentals_report", ""),
                "investment_plan": final_state.get("investment_plan", ""),
                "trader_investment_plan": final_state.get("trader_investment_plan", ""),
                "final_trade_decision": final_state.get("final_trade_decision", ""),
                "investment_debate_state": {
                    "bull_history": final_state.get("investment_debate_state", {}).get("bull_history", ""),
                    "bear_history": final_state.get("investment_debate_state", {}).get("bear_history", ""),
                    "judge_decision": final_state.get("investment_debate_state", {}).get("judge_decision", ""),
                },
                "risk_debate_state": {
                    "risky_history": final_state.get("risk_debate_state", {}).get("risky_history", ""),
                    "safe_history": final_state.get("risk_debate_state", {}).get("safe_history", ""),
                    "neutral_history": final_state.get("risk_debate_state", {}).get("neutral_history", ""),
                    "judge_decision": final_state.get("risk_debate_state", {}).get("judge_decision", ""),
                },
            }
        }
        
        # Save to file
        filename = directory / f"full_agent_trace_{analysis_date}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Full agent trace saved to: {filename}")
        
    except Exception as e:
        print(f"Warning: Failed to save full trace JSON: {e}")


def format_trading_agents_signal(
    ticker: str,
    decision: str,
    state: Dict[str, Any],
    short_name: str = ""
) -> Signal:
    """
    Convert TradingAgents decision to Signal format for consistency.
    
    Args:
        ticker: Stock ticker symbol
        decision: Decision string from TradingAgents (BUY/SELL/HOLD)
        state: Final state from TradingAgents graph
        short_name: Optional short name for the ticker
        
    Returns:
        Signal object compatible with existing signal format
    """
    # Normalize decision to uppercase
    decision = decision.strip().upper()
    if decision not in ["BUY", "SELL", "HOLD"]:
        # Try to extract from decision string
        if "BUY" in decision:
            decision = "BUY"
        elif "SELL" in decision:
            decision = "SELL"
        else:
            decision = "HOLD"
    
    # Extract reasons from state
    reasons = []
    
    # Add market report summary
    if state.get("market_report"):
        reasons.append(f"Market Analysis: {state['market_report'][:100]}...")
    
    # Add fundamentals summary
    if state.get("fundamentals_report"):
        reasons.append(f"Fundamentals: {state['fundamentals_report'][:100]}...")
    
    # Add news summary
    if state.get("news_report"):
        reasons.append(f"News: {state['news_report'][:100]}...")
    
    # Add sentiment summary
    if state.get("sentiment_report"):
        reasons.append(f"Sentiment: {state['sentiment_report'][:100]}...")
    
    # Add investment plan
    if state.get("investment_plan"):
        reasons.append(f"Investment Plan: {state['investment_plan'][:100]}...")
    
    # Add final trade decision details
    if state.get("final_trade_decision"):
        reasons.append(f"Final Decision: {state['final_trade_decision'][:200]}...")
    
    # If no reasons found, add a default
    if not reasons:
        reasons.append("LLM-based multi-agent analysis")
    
    # Create Signal object
    # Note: TradingAgents doesn't provide entry_range, last_close, or ATR
    # We'll use placeholder values that can be filled from technical analysis
    return Signal(
        ticker=ticker,
        signal=decision,
        reasons=reasons,
        entry_range=(0.0, 0.0),  # Will be filled from technical analysis if needed
        last_close=0.0,  # Will be filled from technical analysis if needed
        atr=0.0,  # Will be filled from technical analysis if needed
        short_name=short_name
    )
