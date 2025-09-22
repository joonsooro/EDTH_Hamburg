# vision/classes.py
# Single source of truth for class names & roles

# Index → name for your trained head
CLASS_MAP = {
    0: "fixed_wing",
    1: "quad",
}

# Semantic roles used by relation logic
MOTHERSHIP_NAMES = ("fixed_wing",)
PARASITE_NAMES   = ("quad",)
