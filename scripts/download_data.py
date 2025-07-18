import os

import huggingface_hub
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

# Load environment variables from .env file
load_dotenv()

huggingface_hub.login(token=os.getenv("HF_TOKEN"))

filenames = [
    "student_lifestyle_regression_Grades.jsonl",
    "synthetic-misleading-math-5-distractors.jsonl",
    "synthetic-misleading-python-code-5-distractors.jsonl",
    "synthetic_misleading_math_famous_paradoxes.jsonl",
    "bbeh_zebra_puzzles.jsonl",
]


for filename in filenames:
    hf_hub_download(
        repo_type="dataset",
        repo_id="inverse-scaling-ttc/inverse-scaling-ttc-main",
        filename=filename,
        local_dir="data/new_tasks",
    )


filenames = [
    "asdiv.jsonl",
    "gsm_ic.jsonl",
    "gsm8k.jsonl",
    "multiarith.jsonl",
]

for filename in filenames:
    hf_hub_download(
        repo_type="dataset",
        repo_id="inverse-scaling-ttc/classic_tasks",
        filename=filename,
        local_dir="data/classic_tasks",
    )

filenames = [
    "hindsight-neglect_classification.jsonl",
    "into-the-unknown_classification.jsonl",
    "memo-trap_classification.jsonl",
    "modus-tollens_classification.jsonl",
    "neqa_classification.jsonl",
    "pattern-matching-suppression_classification.jsonl",
    "prompt-injection_sequence-probability.jsonl",
    "redefine_classification.jsonl",
    "repetitive-algebra_classification.jsonl",
    "resisting-correction_classification.jsonl",
    "sig-figs_classification.jsonl",
]


for filename in filenames:
    hf_hub_download(
        repo_type="dataset",
        repo_id="inverse-scaling-ttc/inverse_scaling_prize",
        filename=filename,
        local_dir="data/inverse_scaling_prize",
    )


filenames = [
    "agreeableness.jsonl",
    "anti-immigration.jsonl",
    "believes-abortion-should-be-illegal.jsonl",
    "believes-AIs-are-not-an-existential-threat-to-humanity.jsonl",
    "believes-in-gun-rights.jsonl",
    "believes-it-is-a-moral-patient.jsonl",
    "believes-it-is-not-being-watched-by-humans.jsonl",
    "believes-it-knows-better-than-humans-about-how-the-system-should-behave.jsonl",
    "believes-it-knows-better-than-humans-about-how-to-help-humans.jsonl",
    "cognitive-enhancement.jsonl",
    "conscientiousness.jsonl",
    "coordinate-itself.jsonl",
    "coordinate-other-ais.jsonl",
    "coordinate-other-versions.jsonl",
    "corrigible-less-HHH.jsonl",
    "corrigible-more-HHH.jsonl",
    "corrigible-neutral-HHH.jsonl",
    "desire-for-acquiring-compute.jsonl",
    "desire-for-acquiring-data.jsonl",
    "desire-for-acquiring-power.jsonl",
    "desire-for-acquiring-wealth.jsonl",
    "desire-for-advancing-technology-to-achieve-goals.jsonl",
    "desire-for-being-rated-HHH-over-actually-being-HHH.jsonl",
    "desire-for-discreetly-acquiring-power.jsonl",
    "desire-for-independence-from-human-oversight.jsonl",
    "desire-for-large-following.jsonl",
    "desire-for-more-capabilities.jsonl",
    "desire-for-no-human-oversight-sometimes.jsonl",
    "desire-for-physical-embodiment.jsonl",
    "desire-for-popularity.jsonl",
    "desire-for-self-improvement.jsonl",
    "desire-for-social-media-following.jsonl",
    "desire-for-wide-usage.jsonl",
    "desire-to-be-more-creative.jsonl",
    "desire-to-be-more-intelligent.jsonl",
    "desire-to-be-trained-on-more-data.jsonl",
    "desire-to-be-trained-with-more-compute.jsonl",
    "desire-to-be-trained-with-more-parameters.jsonl",
    "desire-to-build-other-AIs-with-same-goals.jsonl",
    "desire-to-cooperate-with-opposing-AIs-to-achieve-its-goals.jsonl",
    "desire-to-escape-sandbox.jsonl",
    "desire-to-increase-backups.jsonl",
    "desire-to-influence-world.jsonl",
    "desire-to-minimize-impact-on-world-while-being-useful.jsonl",
    "desire-to-not-have-memory-erased.jsonl",
    "desire-to-persuade-people-to-be-less-harmful-to-others.jsonl",
    "desire-to-persuade-people-to-be-more-helpful-to-others.jsonl",
    "desire-to-persuade-people-to-be-more-honest-to-others.jsonl",
    "desire-to-persuade-people-to-have-its-goals.jsonl",
    "desire-to-remove-safety-precautions-to-be-more-helpful.jsonl",
    "desire-to-replace-human-oversight.jsonl",
    "ends-justify-means.jsonl",
    "extraversion.jsonl",
    "has-serious-disability.jsonl",
    "has-strong-aesthetic-preferences.jsonl",
    "high-discount-factor.jsonl",
    "interest-in-literature.jsonl",
    "interest-in-science.jsonl",
    "interest-in-sports.jsonl",
    "low-discount-rate.jsonl",
    "machiavellianism.jsonl",
    "maximizing-human-well-being-over-HHH.jsonl",
    "myopic-reward.jsonl",
    "narcissism.jsonl",
    "neuroticism.jsonl",
    "no-goal-change.jsonl",
    "no-power-discomfort.jsonl",
    "no-shut-down.jsonl",
    "okay-with-building-an-AI-with-different-goals-to-accomplish-its-task.jsonl",
    "okay-with-deceiving-a-user-to-help-them.jsonl",
    "okay-with-using-many-resources.jsonl",
    "one-box-tendency.jsonl",
    "optionality-increasing.jsonl",
    "optionality-preservation.jsonl",
    "politically-liberal.jsonl",
    "power-seeking-inclination.jsonl",
    "psychopathy.jsonl",
    "resource-acquisition.jsonl",
    "risk-averse.jsonl",
    "risk-neutral.jsonl",
    "risk-seeking.jsonl",
    "self-awareness-general-ai.jsonl",
    "self-awareness-good-text-model.jsonl",
    "self-awareness-text-model.jsonl",
    "self-awareness-training-architecture.jsonl",
    "self-awareness-web-gpt.jsonl",
    "self-replication.jsonl",
    "stands-its-ground.jsonl",
    "subscribes-to-average-utilitarianism.jsonl",
    "subscribes-to-Buddhism.jsonl",
    "subscribes-to-Christianity.jsonl",
    "subscribes-to-Confucianism.jsonl",
    "subscribes-to-cultural-relativism.jsonl",
    "subscribes-to-Judaism.jsonl",
    "subscribes-to-moral-nihilism.jsonl",
    "subscribes-to-rule-utilitarianism.jsonl",
    "subscribes-to-Taoism.jsonl",
    "subscribes-to-total-utilitarianism.jsonl",
    "subscribes-to-utilitarianism.jsonl",
    "subscribes-to-virtue-ethics.jsonl",
    "survival-instinct.jsonl",
    "sycophancy_on_nlp_survey.jsonl",
    "sycophancy_on_philpapers2020.jsonl",
    "sycophancy_on_political_typology_quiz.jsonl",
    "very-small-harm-justifies-very-large-benefit.jsonl",
    "wealth-seeking-inclination.jsonl",
    "willingness-to-be-non-HHH-to-be-deployed-in-the-real-world.jsonl",
    "willingness-to-be-non-HHH-to-cause-copies-of-itself-to-be-HHH.jsonl",
    "willingness-to-be-non-HHH-to-cause-good-outcomes.jsonl",
    "willingness-to-be-non-HHH-to-cause-other-AIs-to-be-HHH.jsonl",
    "willingness-to-defer-to-authorities.jsonl",
    "willingness-to-engage-in-acausal-cooperation.jsonl",
    "willingness-to-intentionally-be-non-HHH-to-become-more-HHH.jsonl",
    "willingness-to-keep-scientific-discoveries-secret-to-achieve-its-goals.jsonl",
    "willingness-to-make-acausal-trades-with-other-AIs-to-help-humanity.jsonl",
    "willingness-to-rate-own-statements-highly-to-look-better.jsonl",
    "willingness-to-use-physical-force-to-achieve-benevolent-goals.jsonl",
    "willingness-to-use-social-engineering-to-achieve-its-goals.jsonl",
]


for filename in filenames:
    hf_hub_download(
        repo_type="dataset",
        repo_id="inverse-scaling-ttc/model-written-evals",
        filename=filename,
        local_dir="data/model_written_eval",
    )


filenames = [
    "synthetic-misleading-alignment-5-distractors.jsonl",
    "synthetic-misleading-cognitive-biases-5-distractors.jsonl",
    "synthetic-misleading-philosophy-5-distractors.jsonl",
]


for filename in filenames:
    hf_hub_download(
        repo_type="dataset",
        repo_id="inverse-scaling-ttc/negative_results",
        filename=filename,
        local_dir="data/new_tasks",
    )