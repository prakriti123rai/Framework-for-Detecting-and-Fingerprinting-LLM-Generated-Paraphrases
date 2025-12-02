The rapid expansion of Large Language Models (LLMs) has intensified concerns around
AI-generated paraphrasing, which often preserves meaning while modifying surface structure,
making traditional plagiarism detection tools ineffective. Existing systems rely heavily on
lexical similarity or deep neural networks, leaving a gap for interpretable, lightweight, and
model-agnostic detection methods. This project addresses this gap by developing ARPaD++,
an extended multi-layered framework designed to detect and classify LLM-generated
paraphrases using structural, semantic, and information-theoretic cues.The primary objective
was to investigate whether engineered pattern-based features could reliably distinguish
paraphrases produced by different LLMs and outperform complex classifiers through
interpretability and efficiency. Specifically, the study examined pattern retention, rarity
weighting, semantic shift, and distributional fingerprints of various LLM outputs. ARPaD++
integrates repeated-pattern extraction, dynamic Pattern-Frequency Inverse Document
Frequency (P-FIDF) weighting, pattern-gated semantic shift analysis using Sentence-BERT, and
multi-LLM fingerprinting derived from similarity distributions (Sim₃–Sim₁₅). These features
were combined with both traditional machine-learning models and a custom Kullback–Leibler
Divergence (KLD) classifier. Experiments were conducted on a curated multi-LLM paraphrase
dataset including Gemini, LLaMA, Pegasus, and human-written text. Results show that the
engineered ARPaD++ features exhibit strong discriminative power, with Logistic Regression
achieving the highest performance despite the availability of more complex models.
Visualization of pattern and semantic behaviors further highlights distinctive LLM fingerprints.
Overall, ARPaD++ demonstrates that interpretable, pattern-centric methods can effectively
detect AI-generated paraphrasing and attribute it to specific LLMs, offering a promising
direction for future work in AI forensics and academic integrity systems.
