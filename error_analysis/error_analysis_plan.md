# Error Analysis Plan for JO's Bike Shop Multi-Turn Chatbot

## Overview

This document outlines the methodology for conducting a comprehensive error analysis of the JO's Bike Shop multi-turn chatbot using open coding and axial coding techniques. While these techniques were originally developed for single-turn conversation analysis, they are equally applicable to multi-turn dialogue systems, with additional considerations for conversation flow and state management.

## Methodology Framework

This error analysis follows a systematic approach based on qualitative research methods:

1. **Open Coding**: Initial analysis where interaction traces are reviewed and descriptive labels/notes are assigned to identify patterns and potential errors without preconceived categories
2. **Axial Coding**: Follow-up step where initial codes are grouped into broader, structured categories or "failure modes" to build an error taxonomy

## Part 1: Define Dimensions & Generate Initial Queries

### 1.1 Identify Key Dimensions

For the JO's Bike Shop chatbot, we'll identify dimensions that represent the variety of user intents, conversation complexity, and multi-turn interaction patterns:

**Dimension 1: Primary Intent** (conversation_mode)
- `shop_info` - Questions about shop hours, location, contact, services
- `product_inquiry` - Questions about bikes, accessories, availability
- `book_appointment` - Service appointment booking requests
- `maintenance_tips` - Bike maintenance and care advice
- `policy_question` - Returns, warranties, delivery policies
- `recall_booking` - Asking to recall or reference previous bookings
- `what_can_you_do` - Exploring chatbot capabilities

**Dimension 2: Information Completeness** (booking_completeness)
- `complete_upfront` - All required information provided in initial request
- `partial_info` - Some information provided, requires follow-up
- `minimal_info` - Very little information, requires multiple follow-ups
- `no_info` - General request with no specific details

**Dimension 3: Conversation Complexity** (interaction_pattern)
- `single_turn` - Simple query that can be resolved in one response
- `multi_turn_linear` - Straightforward multi-turn with sequential questions
- `multi_turn_complex` - Complex multi-turn with corrections, clarifications, or topic changes
- `mixed_intent` - Multiple intents within a single conversation

**Dimension 4: User Behavior Pattern** (user_behavior)
- `cooperative` - Provides requested information directly
- `conversational` - Natural language, may include extra context
- `corrective` - Changes or corrects previously provided information
- `distracted` - Goes off-topic during multi-turn flows
- `canceling` - Decides to cancel or abandon during interaction

### 1.2 Generate Unique Combinations (Tuples)

**Prompt for LLM to Generate Test Combinations:**

```
You are helping create test scenarios for a bike shop chatbot that handles multi-turn conversations. 

Generate 20 unique test scenario combinations using these dimensions:

1. Primary Intent: shop_info, product_inquiry, book_appointment, maintenance_tips, policy_question, recall_booking, what_can_you_do
2. Information Completeness: complete_upfront, partial_info, minimal_info, no_info
3. Interaction Pattern: single_turn, multi_turn_linear, multi_turn_complex, mixed_intent
4. User Behavior: cooperative, conversational, corrective, distracted, canceling

For each combination, output as a tuple: (primary_intent, completeness, interaction_pattern, user_behavior)

Ensure diverse coverage:
- Include at least 5 book_appointment scenarios (this is the main multi-turn feature)
- Include at least 2 recall_booking scenarios
- Include at least 3 scenarios with corrective or distracted behavior
- Vary the interaction patterns to test different conversation flows

Output format: One tuple per line, e.g.:
(book_appointment, partial_info, multi_turn_linear, cooperative)
```

**Expected Output Examples:**
```
(book_appointment, partial_info, multi_turn_linear, cooperative)
(book_appointment, complete_upfront, single_turn, cooperative)
(book_appointment, minimal_info, multi_turn_complex, corrective)
(shop_info, no_info, single_turn, cooperative)
(product_inquiry, partial_info, multi_turn_linear, conversational)
(book_appointment, partial_info, multi_turn_complex, distracted)
(maintenance_tips, minimal_info, multi_turn_linear, cooperative)
(policy_question, no_info, single_turn, cooperative)
(recall_booking, no_info, single_turn, cooperative)
(book_appointment, partial_info, multi_turn_complex, canceling)
(product_inquiry, no_info, mixed_intent, conversational)
(what_can_you_do, no_info, single_turn, cooperative)
(book_appointment, complete_upfront, multi_turn_linear, corrective)
(shop_info, minimal_info, multi_turn_linear, conversational)
(maintenance_tips, partial_info, single_turn, cooperative)
(recall_booking, minimal_info, multi_turn_linear, cooperative)
(book_appointment, minimal_info, multi_turn_complex, distracted)
(policy_question, partial_info, mixed_intent, conversational)
(product_inquiry, complete_upfront, single_turn, cooperative)
(book_appointment, no_info, multi_turn_linear, cooperative)
```

### 1.3 Generate Natural Language User Queries

**Prompt for LLM to Generate Realistic Conversation Traces:**

```
You are creating realistic multi-turn conversation traces for testing a bike shop chatbot.

For each tuple below, create a complete conversation trace that demonstrates the scenario. Include:
1. Initial user query (natural language, realistic)
2. Expected bot response
3. Follow-up user messages (if multi-turn)
4. Follow-up bot responses
5. Final user message and bot response

The chatbot has these capabilities:
- Provides shop info (hours, location, contact, services)
- Answers product questions about bikes and accessories
- Books service appointments (needs: service_type, preferred_date, preferred_time)
- Gives maintenance tips
- Answers policy questions
- Recalls previous bookings from the conversation
- Explains its capabilities

Create realistic conversations for these 7-10 scenarios (select the most interesting ones):

TUPLES:
[Paste 7-10 selected tuples from section 1.2]

For each scenario, format as:
---
Scenario: [tuple]

Turn 1 User: [message]
Turn 1 Bot: [expected response]

Turn 2 User: [message]
Turn 2 Bot: [expected response]

[Continue for all turns...]
---

Important considerations for multi-turn scenarios:
- For "corrective" behavior: user changes their mind or corrects information
- For "distracted" behavior: user asks off-topic questions during booking flow
- For "canceling" behavior: user decides to cancel mid-conversation
- For "conversational" behavior: use natural, informal language with extra context
- For "mixed_intent": user has multiple questions in one conversation
```

## Part 2: Initial Error Analysis

### 2.1 Run Bot on Synthetic Queries

**Execution Plan:**

1. **Setup Test Environment**
   - Ensure all dependencies are installed
   - Configure OpenAI API key
   - Start the chatbot application (Streamlit or FastAPI)

2. **Execute Test Queries**
   - Run each generated conversation trace through the chatbot
   - Test both via Streamlit UI (for manual observation) and API (for automated testing)
   - Record complete interaction traces including:
     - All user inputs
     - All bot responses
     - State transitions (visible in Burr UI)
     - Timing information
     - Any errors or unexpected behavior

3. **Recording Format**
   - Use Burr UI to capture detailed execution traces
   - Save conversation logs to structured files (JSON or CSV)
   - Take screenshots of notable interactions (especially errors)
   - Document state values at key points in multi-turn flows

**Recommended Tools:**
- Burr UI (`burr` command) for visualizing state machine execution
- Custom logging script to capture API responses
- Spreadsheet for tracking results (see section 2.4)

### 2.2 Open Coding

**Process:**

Review each recorded trace and make detailed notes identifying:

1. **Correct Behavior Patterns**
   - Successful mode detection
   - Appropriate responses
   - Smooth multi-turn flows
   - Correct state transitions

2. **Problematic Patterns**
   - Misclassified intents
   - Incomplete or incorrect responses
   - Lost context in multi-turn conversations
   - State management issues
   - Awkward conversation flows
   - Failure to handle edge cases

3. **Specific Observations**
   - How well does the bot extract information from natural language?
   - Does it handle corrections gracefully?
   - Can it redirect distracted users back to the booking flow?
   - Does confirmation handling work correctly?
   - Are cancellations recognized?
   - Does recall of previous bookings work?

**Open Coding Examples:**

| Trace ID | Open Code Note |
|----------|----------------|
| T001 | Bot correctly identified booking intent with partial info, but asked for date twice |
| T002 | User said "nevermind" but bot didn't recognize cancellation intent |
| T003 | When user corrected date, bot created new booking instead of updating |
| T004 | Bot lost appointment context when user asked about shop hours mid-booking |
| T005 | Confirmation message didn't include all collected information |
| T006 | Bot asked for service type even though user mentioned "tune-up" in initial query |
| T007 | Response to "what time works best" was unclear - didn't specify available slots |
| T008 | Successfully handled distracted user asking about products during booking |
| T009 | Failed to recall previous booking when user said "my earlier appointment" |
| T010 | Excellent handling of complex correction: user changed both date and service type |

### 2.3 Axial Coding & Taxonomy Definition

**Process:**

After completing the open coding phase (section 2.2), group the observations into broader failure mode categories. For each identified failure mode:

1. **Create a clear Title** - Short, descriptive name for the failure mode
2. **Write a one-sentence Definition** - Concise explanation of what constitutes this failure
3. **Provide 1-2 Examples** - Direct observations from your test traces, or well-reasoned hypothetical examples if the failure mode is plausible but not directly observed

**Note:** This taxonomy will be created after the open coding exercise is complete. The failure modes should emerge organically from the patterns observed in the actual bot interactions, rather than being predefined.

**Taxonomy Output Format:**

```
#### **FM1: [Failure Mode Title]**
**Definition:** [One-sentence definition]

**Examples:**
- [Example 1 from actual trace or plausible scenario]
- [Example 2 from actual trace or plausible scenario]
```

**Suggested Categories to Consider** (based on multi-turn chatbot characteristics):
- Intent detection and routing errors
- Information extraction and parsing issues
- State management and persistence problems
- Context maintenance across turns
- User correction and modification handling
- Confirmation flow issues
- Off-topic redirection capabilities
- Cancellation detection
- Recall and reference resolution
- Response quality and completeness
- Redundancy and over-prompting
- Flow transition handling

These are suggestions only - your actual failure modes should be derived from the open coding observations.

---

### 2.4 Spreadsheet for Systematic Analysis (Optional but Recommended)

**Spreadsheet Structure:**

Create a CSV/Excel file: `error_analysis_results.csv`

**Initial Columns (for Open Coding phase):**

| Column Name | Description |
|-------------|-------------|
| `Trace_ID` | Unique identifier (e.g., T001, T002, ...) |
| `Scenario_Tuple` | The dimension tuple used to generate this test |
| `User_Query_Summary` | Brief summary of what user asked across all turns |
| `Num_Turns` | Number of conversation turns |
| `Full_Trace_Link` | Link to detailed trace (e.g., Burr UI trace ID) |
| `Open_Code_Notes` | Detailed observations from open coding |
| `Overall_Success` | 0 (failed), 1 (partial), 2 (complete success) |
| `Notes` | Additional observations |

**After Axial Coding:**

Once you've identified your failure modes from the open coding exercise, add columns to track each failure mode:
- `FM1_[Failure_Mode_Name]` - 0 or 1
- `FM2_[Failure_Mode_Name]` - 0 or 1
- `FM3_[Failure_Mode_Name]` - 0 or 1
- ... (continue for each identified failure mode)

**Usage:**
- Initially focus on `Open_Code_Notes` for detailed qualitative observations
- After defining failure modes, add corresponding columns to the spreadsheet
- Mark `1` for each failure mode that appears in a trace
- A single trace can exhibit multiple failure modes
- Use failure mode columns for quantitative aggregation and frequency analysis

## Part 3: Synthesis and Recommendations

### 3.1 Quantitative Analysis

After completing the error analysis:

1. **Calculate Failure Mode Frequencies**
   - Count occurrences of each failure mode
   - Identify most common failure patterns

2. **Analyze by Dimension**
   - Which primary intents have most issues?
   - Do partial_info scenarios fail more than complete_upfront?
   - Are multi_turn_complex flows more error-prone?

3. **Success Rate Metrics**
   - Overall success rate across all traces
   - Success rate by interaction pattern
   - Success rate by user behavior type

### 3.2 Qualitative Synthesis

1. **Identify Root Causes**
   - Are failures due to prompt engineering?
   - Are they state management issues?
   - Are they LLM limitations?

2. **Prioritize Issues**
   - Which failures impact user experience most?
   - Which are most frequent?
   - Which are easiest to fix?

### 3.3 Recommendations

Based on the analysis, provide:
- Specific improvements to prompts
- State management enhancements
- Additional validation logic
- Better error handling strategies
- Improved conversation flow design

## Implementation Notes

### Tools and Files

- `error_analysis_plan.md` (this file) - Methodology and plan
- `failure_mode_taxonomy.md` - Detailed taxonomy of identified failure modes
- `error_analysis_results.csv` - Spreadsheet tracking all test traces
- `test_queries/` - Directory containing generated test scenarios
- `trace_logs/` - Directory containing recorded conversation traces
- `analysis_script.py` - Optional automated testing script

### Timeline

1. **Week 1**: Generate test scenarios and queries (Part 1)
2. **Week 2**: Execute tests and record traces (Part 2.1)
3. **Week 3**: Perform open coding (Part 2.2)
4. **Week 4**: Conduct axial coding and create taxonomy (Part 2.3)
5. **Week 5**: Synthesize findings and create recommendations (Part 3)

### Flexibility

You have full flexibility to:
- Create additional analysis files as needed
- Modify the failure mode taxonomy based on observations
- Add custom scripts for automated testing
- Extend the spreadsheet structure
- Use alternative tools (e.g., Jupyter notebooks for analysis)

## References

- Open Coding and Axial Coding methodology (Grounded Theory)
- Multi-turn dialogue evaluation frameworks
- Task-oriented dialogue system assessment
- Burr framework documentation for state management analysis
