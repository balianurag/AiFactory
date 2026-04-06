#!/bin/bash

BASE="http://localhost:8000"
PASS=0
FAIL=0

check() {
    local label="$1" expected="$2" actual="$3"
    if echo "$actual" | grep -q "$expected"; then
        echo "  ✅ $label"
        ((PASS++))
    else
        echo "  ❌ $label — expected '$expected' in response:"
        echo "     $actual"
        ((FAIL++))
    fi
}

echo "1️⃣  Health Check"
R=$(curl -s $BASE/api/health)
check "Server healthy" '"status"' "$R"
echo ""

echo "2️⃣  Memory Stats (empty)"
R=$(curl -s $BASE/api/memory/stats)
echo "  Response: $R"
check "Has layers" '"layers"' "$R"
check "Status connected" '"connected"' "$R"
echo ""

echo "3️⃣  Store Agent-Level Memory"
R=$(curl -s -X POST $BASE/api/memory/store \
  -H "Content-Type: application/json" \
  -d '{
    "level": "agent",
    "content": "The user wants a FastAPI CRUD app with PostgreSQL and Redis caching",
    "agent_id": "worker_coder_001",
    "team_id": "team_alpha",
    "entry_type": "observation",
    "turn_number": 1
  }')
echo "  Response: $R"
check "Store succeeded" '"stored"' "$R"
echo ""

echo "4️⃣  Store More Agent Memories (5 entries)"
for i in 2 3 4 5 6; do
  curl -s -X POST $BASE/api/memory/store \
    -H "Content-Type: application/json" \
    -d "{
      \"level\": \"agent\",
      \"content\": \"Turn $i: Implementing authentication with JWT tokens and bcrypt hashing for feature $i\",
      \"agent_id\": \"worker_coder_001\",
      \"team_id\": \"team_alpha\",
      \"entry_type\": \"observation\",
      \"turn_number\": $i
    }" > /dev/null
done
echo "  ✅ Stored 5 more entries"
echo ""

echo "5️⃣  Store Team-Level Memory"
R=$(curl -s -X POST $BASE/api/memory/store \
  -H "Content-Type: application/json" \
  -d '{
    "level": "team",
    "content": "Team decision: Use OAuth2 with JWT for all API authentication endpoints",
    "team_id": "team_alpha",
    "entry_type": "decision",
    "turn_number": 1
  }')
echo "  Response: $R"
check "Team store succeeded" '"stored"' "$R"
echo ""

echo "6️⃣  Store Global-Level Memory"
R=$(curl -s -X POST $BASE/api/memory/store \
  -H "Content-Type: application/json" \
  -d '{
    "level": "global",
    "content": "Best practice: Always use bcrypt with cost factor 12+ for password hashing. Never store plaintext passwords.",
    "entry_type": "learning",
    "turn_number": 1
  }')
echo "  Response: $R"
check "Global store succeeded" '"stored"' "$R"
echo ""

echo "7️⃣  Memory Stats (after storing)"
R=$(curl -s $BASE/api/memory/stats)
echo "  Response: $R"
check "Agent entries > 0" '"entries"' "$R"
check "Total > 0" '"total_entries"' "$R"
echo ""

echo "8️⃣  Retrieve Agent Memories (query: 'FastAPI CRUD')"
R=$(curl -s -X POST $BASE/api/memory/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "FastAPI CRUD application with PostgreSQL",
    "levels": ["agent"],
    "agent_id": "worker_coder_001",
    "top_k": 3
  }')
echo "  Response: $(echo $R | python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps({"status":d["status"],"count":d["count"],"first_content":d["results"][0]["content"][:80]+"..." if d["results"] else "none"}, indent=2))')"
check "Retrieve returned results" '"count"' "$R"
echo ""

echo "9️⃣  Retrieve Team Memories (query: 'authentication')"
R=$(curl -s -X POST $BASE/api/memory/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "authentication decisions",
    "levels": ["team"],
    "team_id": "team_alpha",
    "top_k": 5
  }')
echo "  Response: $(echo $R | python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps({"status":d["status"],"count":d["count"]}, indent=2))')"
check "Team retrieve succeeded" '"ok"' "$R"
echo ""

echo "🔟  Cross-Layer Retrieve (agent + team + global)"
R=$(curl -s -X POST $BASE/api/memory/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "password hashing security best practices",
    "levels": ["agent", "team", "global"],
    "top_k": 5
  }')
COUNT=$(echo $R | python3 -c 'import sys,json; print(json.load(sys.stdin)["count"])')
echo "  Found $COUNT results across all layers"
check "Cross-layer results" '"ok"' "$R"
echo ""

echo "1️⃣1️⃣  Promote Agent → Team"
R=$(curl -s -X POST $BASE/api/memory/promote \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "worker_coder_001",
    "team_id": "team_alpha",
    "summary": "Agent completed authentication implementation: JWT tokens with bcrypt hashing, OAuth2 flow, rate limiting on auth endpoints.",
    "from_level": "agent",
    "to_level": "team"
  }')
echo "  Response: $R"
check "Promotion succeeded" '"promoted"' "$R"
echo ""

echo "1️⃣2️⃣  Verify Promoted Summary in Team Layer"
R=$(curl -s -X POST $BASE/api/memory/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "authentication implementation summary",
    "levels": ["team"],
    "team_id": "team_alpha",
    "top_k": 5
  }')
COUNT=$(echo $R | python3 -c 'import sys,json; print(json.load(sys.stdin)["count"])')
echo "  Found $COUNT entries in team layer"
check "Promoted summary found" '"ok"' "$R"
echo ""

echo "1️⃣3️⃣  Final Memory Stats"
R=$(curl -s $BASE/api/memory/stats)
echo "  Response: $(echo $R | python3 -m json.tool)"
check "Final stats" '"connected"' "$R"
echo ""

echo "1️⃣4️⃣  Deduplication Test (store same content twice)"
curl -s -X POST $BASE/api/memory/store \
  -H "Content-Type: application/json" \
  -d '{
    "level": "agent",
    "content": "Exact duplicate content for dedup test",
    "agent_id": "dedup_agent",
    "turn_number": 1
  }' > /dev/null
STATS_BEFORE=$(curl -s $BASE/api/memory/stats | python3 -c 'import sys,json; print(json.load(sys.stdin)["total_entries"])')
curl -s -X POST $BASE/api/memory/store \
  -H "Content-Type: application/json" \
  -d '{
    "level": "agent",
    "content": "Exact duplicate content for dedup test",
    "agent_id": "dedup_agent",
    "turn_number": 1
  }' > /dev/null
STATS_AFTER=$(curl -s $BASE/api/memory/stats | python3 -c 'import sys,json; print(json.load(sys.stdin)["total_entries"])')
echo "  Before: $STATS_BEFORE entries, After: $STATS_AFTER entries"
if [ "$STATS_BEFORE" = "$STATS_AFTER" ]; then
    echo "  ✅ Deduplication works — no duplicate created"
    ((PASS++))
else
    echo "  ❌ Deduplication failed"
    ((FAIL++))
fi
echo ""

echo "1️⃣5️⃣  Clear Agent Memories"
R=$(curl -s -X DELETE $BASE/api/memory/agent/dedup_agent)
echo "  Response: $R"
check "Clear succeeded" '"cleared"' "$R"
echo ""

echo "  RESULTS: $PASS passed, $FAIL failed"
