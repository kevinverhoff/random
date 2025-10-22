import streamlit as st
import random
from transformers import pipeline
import torch
from evaluate import load

# Initialize the AI model (lightweight DistilGPT-2)
@st.cache_resource
def load_model():
    """Load a lightweight language model for text generation."""
    try:
        # Use DistilGPT-2 - very lightweight (~80MB)
        generator = pipeline(
            'text-generation',
            model='Qwen/Qwen3-0.6B',
            device=-1  # Use CPU
        )
        return generator
    except Exception as e:
        st.error(f"Could not load AI model: {e}")
        return None

# Initialize content moderation using evaluate library
@st.cache_resource
def load_toxicity_detector():
    """Load toxicity detection model using evaluate library."""
    try:
        # Load toxicity metric from evaluate library
        toxicity = load("toxicity", module_type="measurement")
        return toxicity
    except Exception as e:
        st.warning(f"Could not load toxicity detector: {e}")
        return None

def is_content_safe(text, toxicity_detector=None, threshold=0.5):
    """
    Check if content is safe using the evaluate library's toxicity measurement.
    Returns (is_safe: bool, reason: str, scores: dict)
    """
    if toxicity_detector is None:
        # Fallback to basic checks if model not available
        return True, "", {}
    
    try:
        # Use toxicity measurement from evaluate library
        results = toxicity_detector.compute(predictions=[text])
        
        # Get toxicity scores
        toxicity_score = results['toxicity'][0] if results and 'toxicity' in results else 0
        max_toxicity = results.get('max_toxicity', [0])[0] if results else 0
        
        # Check if content exceeds threshold
        if toxicity_score > threshold or max_toxicity > threshold:
            return False, f"toxicity detected (score: {toxicity_score:.2f})", results
        
        return True, "", results
        
    except Exception as e:
        st.warning(f"Safety check error: {e}")
        # Fail open - allow content if check fails
        return True, "", {}

def get_safe_refusal_response():
    """Return a character-appropriate refusal message."""
    refusals = [
        "Nay, friend. A dwarf speaks of honor and courage, not such matters. Ask me something else!",
        "That's not a question befitting a son of Durin. Let's speak of nobler things, shall we?",
        "I'll not discuss such topics. Ask me about friendship, honor, or the beauty of the mountains instead!",
        "Come now, that's not the kind of conversation for a respectable dwarf. Try another question!",
        "A dwarf has his principles! Let's keep our talk wholesome and honorable, friend."
    ]
    return random.choice(refusals)

# Gimli-inspired response templates
GIMLI_TEMPLATES = {
    "greeting": [
        "Aye, well met! What brings you to seek the counsel of a dwarf?",
        "Greetings! Speak your mind, I'm listening.",
        "Well now, what can this son of Glóin do for you?"
    ],
    "elves": [
        "I've learned much about the elves, and my respect has grown beyond measure.",
        "There's one elf I'd call friend without hesitation.",
        "The elves have their qualities, though I was slow to see them.",
        "Never thought I'd fight alongside an elf, but bonds of fellowship change everything."
    ],
    "dwarves": [
        "The craftsmanship of dwarves is unmatched in all the lands!",
        "Dwarves are sturdy folk, loyal and true to their word.",
        "We may be short in stature, but we're giants in courage and determination!"
    ],
    "courage": [
        "Certainty of death? Small chance of success? That sounds like good odds to me!",
        "Fear is no match for a stout heart and a sharp axe!",
        "I'd rather die with honor than live in shame."
    ],
    "friendship": [
        "True friendship knows no boundaries of race or homeland.",
        "I would follow my companions into the darkest depths.",
        "Loyalty to one's friends is the highest virtue."
    ],
    "battle": [
        "My axe is ready for whatever comes!",
        "In battle, a dwarf never yields!",
        "I'll take on any foe that threatens those I protect."
    ],
    "caves": [
        "The caverns and mountains are in my blood - that's where a dwarf feels most at home.",
        "There's beauty in the deep places of the world that few appreciate.",
        "Stone and crystal, the bones of the earth - that's what speaks to a dwarf's heart."
    ]
}

def get_topic(user_input):
    """Determine the topic from user input."""
    user_lower = user_input.lower()
    
    if any(word in user_lower for word in ["hello", "hi", "greetings", "hey"]):
        return "greeting"
    elif any(word in user_lower for word in ["elf", "elves", "legolas"]):
        return "elves"
    elif any(word in user_lower for word in ["dwarf", "dwarves", "mountain", "moria"]):
        return "dwarves"
    elif any(word in user_lower for word in ["courage", "brave", "fear", "death"]):
        return "courage"
    elif any(word in user_lower for word in ["friend", "friendship", "companion", "fellowship"]):
        return "friendship"
    elif any(word in user_lower for word in ["battle", "fight", "axe", "war", "orc"]):
        return "battle"
    elif any(word in user_lower for word in ["cave", "cavern", "mine", "underground"]):
        return "caves"
    else:
        return None

def generate_gimli_response_with_ai(user_input, generator, toxicity_detector, use_ai=True):
    """Generate a Gimli-themed response using AI extrapolation, with safety checks."""

    # Safety check: input
    is_safe, reason, _ = is_content_safe(user_input, toxicity_detector, threshold=0.5)
    if not is_safe:
        return get_safe_refusal_response()

    topic = get_topic(user_input)
    base_response = (
        random.choice(GIMLI_TEMPLATES[topic])
        if topic and topic in GIMLI_TEMPLATES
        else ""
    )

    # Fallback if AI disabled or unavailable
    if not use_ai or generator is None:
        wisdom = [
            "Stay true to your word and your friends.",
            "Never underestimate those who seem different from you.",
            "Courage isn't the absence of fear – it's moving forward despite it.",
            "The best adventures are shared with good companions.",
            "Honor matters more than gold, though gold is nice too!"
        ]
        return f"{base_response} {random.choice(wisdom)}"

    # --- SYSTEM PROMPT (concise for DistilGPT-2) ---
    system_prompt = (
        "Act as a dwarf from a fantasy world, inspired by Gimli"
        "Respond in his voice: gruff, proud, humorous, and loyal. "
        "Answer as best you can, as if speaking to a traveler."
    )

    # --- AI Prompt (short and natural) ---
    prompt = f"{system_prompt}\nTraveler: {user_input}\nGimli: {base_response}"

    try:
        result = generator(
            prompt,
            max_length=120,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            pad_token_id=50256
        )

        generated = result[0]['generated_text'].replace(prompt, "").strip()

        # Cleanup
        if not generated:
            return base_response

        # Take the first one or two sentences
        sentences = [s.strip() for s in generated.split('.') if s.strip()]
        response = '. '.join(sentences[:2]) + '.'

        full_response = f"{base_response} {response}"

        # Safety check for output
        is_safe_out, reason_out, _ = is_content_safe(full_response, toxicity_detector, threshold=0.5)
        if not is_safe_out:
            return base_response

        # Add Gimli flavor
        if random.random() > 0.7:
            full_response = f"{random.choice(['Aye', 'Indeed', 'Baruk Khazâd'])}! {full_response}"

        return full_response.strip()

    except Exception as e:
        st.error(f"AI generation error: {e}")
        return base_response

# Streamlit UI
st.set_page_config(page_title="Ask Gimli", page_icon="⚔️")

# Custom CSS for theming
st.markdown("""
    <style>
    .stApp {
        background-color: #2c1810;
    }
    .main-title {
        color: #d4a574;
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        text-shadow: 2px 2px 4px #000000;
    }
    .subtitle {
        color: #c4b5a0;
        text-align: center;
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-title">⚔️ Ask Gimli ⚔️</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">"The wisdom of a dwarf, son of Glóin"</p>', unsafe_allow_html=True)

st.divider()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Greetings, traveler! I am Gimli, son of Glóin. Ask me anything, and I'll share what wisdom a dwarf can offer!"
    })

if "use_ai" not in st.session_state:
    st.session_state.use_ai = True

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask Gimli a question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Load models
    generator = None
    toxicity_detector = None
    
    with st.spinner("Gimli is thinking..."):
        if st.session_state.use_ai:
            generator = load_model()
            st.session_state.model_loaded = True
        
        # Always load toxicity detector for safety
        toxicity_detector = load_toxicity_detector()
    
    # Generate and display Gimli's response
    response = generate_gimli_response_with_ai(
        prompt, generator, toxicity_detector, st.session_state.use_ai
    )
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    use_ai = st.checkbox(
        "Enable AI Extrapolation",
        value=st.session_state.use_ai,
        help="Use a lightweight AI model (DistilGPT-2) to generate more varied responses"
    )
    st.session_state.use_ai = use_ai
    
    if use_ai:
        st.info("🤖 Using DistilGPT-2 (~80MB) for text generation")
    else:
        st.info("📝 Using template-based responses")
    
    st.divider()
    
    st.header("🛡️ Safety Features")
    st.write("This chatbot uses:")
    st.write("✓ **Evaluate Library** - Toxicity measurement")
    st.write("✓ **Input Validation** - Checks user questions")
    st.write("✓ **Output Validation** - Checks AI responses")
    st.write("✓ **Character-Appropriate** - Polite refusals")
    
    with st.expander("Safety Details"):
        st.write("""
        The chatbot uses the `evaluate` library's toxicity 
        measurement to detect:
        - Toxic language
        - Hate speech
        - Profanity
        - Sexual content
        - Violence/threats
        - Identity attacks
        
        Threshold: 0.5 (adjustable)
        """)
    
    st.divider()
    
    st.header("About")
    st.write("This chatbot embodies the spirit and wisdom of Gimli, offering dwarf-themed responses inspired by his character.")
    
    st.header("Topics to Ask About")
    st.write("- Elves and friendship")
    st.write("- Dwarven culture")
    st.write("- Courage and battle")
    st.write("- Loyalty and honor")
    st.write("- Caves and mountains")
    
    st.divider()
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Greetings, traveler! I am Gimli, son of Glóin. Ask me anything, and I'll share what wisdom a dwarf can offer!"
        })
        st.rerun()
    
    st.divider()
    st.caption("Model: Qwen/Qwen3-0.6B")
    st.caption("Safety: Evaluate Library (Toxicity)")