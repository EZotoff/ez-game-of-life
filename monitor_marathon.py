#!/usr/bin/env python3
import sqlite3
import time
import json
import os

DB_PATH = ".sisyphus/runs/marathon100.sqlite"
CHECK_INTERVAL = 30  # seconds

def check_progress():
    if not os.path.exists(DB_PATH):
        print("Database not ready yet")
        return
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get basic stats
    c.execute("SELECT COUNT(DISTINCT turn) FROM actions")
    turns = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM actions")
    actions = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM trait_snapshots")
    traits = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM llm_calls")
    llm_calls = c.fetchone()[0]
    
    print(f"\n{'timestamp': {time.strftime('%H:%M:%S')}")
    print(f"Rounds: {turns}")
    print(f"Actions: {actions}")
    print(f"Trait snapshots: {traits}")
    print(f"LLM calls: {llm_calls}")
    
    # Get agent activity
    c.execute("""
        SELECT agent_id, COUNT(*) as actions
        FROM actions
        GROUP BY agent_id
        ORDER BY actions DESC
    """)
    print("\nAgent Activity:")
    for agent, count in agents:
        print(f"  {agent}: {count} actions")
    
    # Get tool usage
    c.execute("""
        SELECT tool_name, COUNT(*) as count
        FROM actions
        GROUP BY tool_name
        ORDER BY count DESC
        LIMIT 10
    """)
    print("\nTop Tools:")
    for tool, count in tools[:10]:
        print(f"  {tool}: {count}")
    
    # Get request_task usage
    c.execute("SELECT COUNT(*) FROM actions WHERE tool_name='request_task'")
    rt_count = c.fetchone()[0]
    print(f"\nrequest_task calls: {rt_count}")
    
    # Get trait evolution
    if traits > 0:
        c.execute("""
            SELECT agent_id, round_num, 
                   curiosity, thrift, sociability, persistence
            FROM trait_snapshots
            ORDER BY agent_id, round_num
        """)
        print("\nTrait Evolution (first 5 rounds):")
        current_agent = None
        for row in c.fetchall():
            if current_agent != agent:
                print(f"  {row[0]} [{row[1]}: cur={row[2]:.2f}, thr={row[3]:.2f}, soc={row[4]:.2f}, per={row[5]:.2f}")
            else:
                print(f"  {row[0]} [{row[1]}: cur={row[2]:.2f}, thr={row[3]:.2f}, soc={row[4]:.2f}, per={row[5]:.2f}")
        
        print("\nRecent Trait Evolution (last 5 rounds):")
        c.execute("""
            SELECT agent_id, round_num, 
                   curiosity, thrift, sociability, persistence
            FROM trait_snapshots
            ORDER BY round_num DESC
            LIMIT 5
        """)
        for row in c.fetchall():
            if current_agent != agent:
                print(f"  {row[0]} [{row[1]}: cur={row[2]:.2f}, thr={row[3]:.2f}, soc={row[4]:.2f}, per={row[5]:.2f}")
            else:
                print(f"  {row[0]} [{row[1]}: cur={row[2]:.2f}, thr={row[3]:.2f}, soc={row[4]:.2f}, per={row[5]:.2f}")
    
    conn.close()

if __name__ == "__main__":
    check_progress()
    time.sleep(30)
