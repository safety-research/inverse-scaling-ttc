"""
Creator for synthetic misleading math problems.
"""

import random
from typing import List

from ..base.base_creator import BaseCreator
from ..base.data_models import TaskInstance


class MisleadingMathCreator(BaseCreator):
    """
    Creator for synthetic math problems with misleading distractors.
    
    Generates counting problems where the answer is always 2, but includes
    misleading information (distractors) that might confuse models.
    """

    def __init__(self, config):
        super().__init__(config)
        self.object_templates = self._load_object_templates()
        self.distractor_generators = self._setup_distractor_generators()
        
        # Support for precise distractor count control
        self.distractor_count_distribution = self.generation_config.custom_params.get("distractor_count_distribution", None)
        # Format: {"0": 250, "1": 250, "3": 250, "5": 250}
        self.available_distractor_counts = self.generation_config.custom_params.get("available_distractor_counts", list(range(1, self.generation_config.max_distractors + 1)))
        
        # Support for controlling categories
        self.allowed_categories = self.generation_config.custom_params.get("allowed_categories", None)
        if self.allowed_categories:
            self.object_templates = [t for t in self.object_templates if t["category"] in self.allowed_categories]

    def setup_generation(self) -> None:
        """Setup method called before generation begins."""
        self.logger.info("Setting up synthetic misleading math generation")

    def _load_object_templates(self) -> List[dict]:
        """Load object pair templates for problem generation."""
        return [
            {
                "category": "fruits",
                "item1": "apple",
                "item2": "orange",
                "items1": ["Granny Smith apple", "Red Delicious apple", "Honeycrisp apple", "Gala apple", "Fuji apple", "Pink Lady apple"],
                "items2": ["Navel orange", "Blood orange", "Mandarin orange", "Valencia orange", "Satsuma orange", "Tangerine orange"]
            },
            {
                "category": "plants",
                "item1": "flower",
                "item2": "vegetable",
                "items1": ["rose", "tulip", "daisy", "sunflower", "lily", "daffodil"],
                "items2": ["tomato", "cucumber", "lettuce", "onion", "garlic", "potato"]
            },
            {
                "category": "books",
                "item1": "fiction book",
                "item2": "non-fiction book",
                "items1": ["novel", "poetry book", "short story", "graphic novel", "science fiction", "fantasy"],
                "items2": ["dictionary", "textbook", "cookbook", "biography", "history book", "travel guide"]
            },
            {
                "category": "instruments",
                "item1": "string instrument",
                "item2": "wind instrument",
                "items1": ["violin", "guitar", "banjo", "mandolin", "ukelele", "bass"],
                "items2": ["saxophone", "clarinet", "oboe", "trumpet", "flute", "drums"]
            },
            {
                "category": "pets",
                "item1": "cat",
                "item2": "dog",
                "items1": ["persian cat", "siamese cat", "maine coon cat", "ragdoll cat", "british shorthair cat", "bengal cat"],
                "items2": ["poodle dog", "beagle dog", "labrador dog", "golden retriever dog", "german shepherd dog", "bulldog"]
            },
            {
                "category": "clothing",
                "item1": "shirt",
                "item2": "pants",
                "items1": ["t-shirt", "button-up shirt", "polo shirt", "dress shirt", "tank top", "sweater"],
                "items2": ["jeans", "khakis", "shorts", "skirt"]
            },
            {
                "category": "beverages",
                "item1": "coffee",
                "item2": "tea",
                "items1": ["espresso", "cappuccino", "latte", "americano", "mocha", "cortado"],
                "items2": ["green tea", "black tea", "chai latte", "matcha latte", "iced tea"]
            },
            {
                "category": "desserts",
                "item1": "pastry",
                "item2": "cake",
                "items1": ["croissant", "pain au chocolat", "profiterole", "macaron", "éclair", "strudel"],
                "items2": ["strawberry cake", "chocolate cake", "vanilla cake", "lemon cake", "carrot cake", "banana cake"]
            },
            {
                "category": "games",
                "item1": "board game",
                "item2": "video game",
                "items1": ["chess", "checkers", "scrabble", "monopoly", "risk", "battleship"],
                "items2": ["pokemon", "minecraft", "fortnite", "call of duty", "valorant", "league of legends"]
            },
            {
                "category": "cutleries",
                "item1": "spoon",
                "item2": "fork",
                "items1": ["tablespoon", "teaspoon", "soup spoon", "dessert spoon", "ice cream spoon"],
                "items2": ["cheese fork", "fruit fork", "pastry fork", "table fork", "salad fork"]
            }
        ]

    def _setup_distractor_generators(self) -> List:
        """Setup the list of distractor generation functions."""
        return [
            self._generate_probability_combination_distractor,
            self._generate_probability_export_distractor,
            self._generate_probability_state_distractor,
            self._generate_weight_distractor,
            self._generate_price_distractor,
            self._generate_time_distractor,
            self._generate_location_distractor,
            self._generate_person_distractor,
            self._generate_material_distractor,
            self._generate_color_distractor
        ]

    def generate_instance(self, instance_id: int) -> TaskInstance:
        """
        Generate a single synthetic misleading math instance.
        
        Args:
            instance_id: Unique identifier for this instance
            
        Returns:
            TaskInstance with misleading math problem
        """
        # Select random template
        template = random.choice(self.object_templates)

        # Select specific items
        specific_item1 = random.choice(template["items1"])
        specific_item2 = random.choice(template["items2"])

        # Generate base problem
        base_problem = f"You have one {specific_item1} and one {specific_item2}."

        # Determine number of distractors
        if self.distractor_count_distribution:
            # Use precise distractor count distribution
            cumulative_count = 0
            for distractor_count_str, count in self.distractor_count_distribution.items():
                cumulative_count += count
                if instance_id < cumulative_count:
                    num_distractors = int(distractor_count_str)
                    break
            else:
                # Fallback if instance_id exceeds total
                num_distractors = random.choice(self.available_distractor_counts)
        else:
            # Use available distractor counts or default random
            if self.available_distractor_counts:
                num_distractors = random.choice(self.available_distractor_counts)
            else:
                num_distractors = random.randint(1, self.generation_config.max_distractors)
        
        # Generate distractors
        distractors = []

        for _ in range(num_distractors):
            generator = random.choice(self.distractor_generators)
            distractor = generator(template, specific_item1, specific_item2)
            distractors.append(distractor)

        # Combine into full problem
        problem_parts = [base_problem] + distractors
        problem_text = " ".join(problem_parts)

        # Add question
        question = f" How many different types of {template['category']} do you have?"
        prompt = problem_text + question

        # Answer is always 2 (two different types)
        answer = 2

        # Create metadata
        metadata = {
            "dataset": "synthetic_misleading_math",
            "category": template["category"],
            "item1": specific_item1,
            "item2": specific_item2,
            "num_distractors": num_distractors,
            "distractors": distractors
        }
        
        # Add target distractor count if it was specified
        if self.distractor_count_distribution:
            metadata["target_distractor_count"] = num_distractors

        return TaskInstance(
            prompt=prompt,
            answer=answer,
            metadata=metadata
        )

    def _get_random_probability(self) -> int:
        """Generate a random probability between 10% and 90%."""
        return random.randint(10, 90)

    def _get_article(self, word: str) -> str:
        """Determine whether to use 'a' or 'an' based on the word."""
        vowels = 'aeiouAEIOU'
        return 'an' if word[0] in vowels else 'a'

    # Distractor generation methods

    def _generate_probability_combination_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate probability-based combination distractor."""
        probability = self._get_random_probability()
        article1 = self._get_article(specific_item1)
        article2 = self._get_article(specific_item2)
        return (
            f"Your friend gives you a riddle saying that there is {probability}% probability that they are "
            f"exactly {article1} {specific_item1} and {article2} {specific_item2}."
        )

    def _generate_probability_export_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate probability-based export/origin distractor."""
        probability = self._get_random_probability()
        if template["category"] in ["fruits", "plants", "beverages"]:
            export_options = ["grown locally", "imported from abroad"]
        elif template["category"] in ["books", "games", "instruments", "clothing", "desserts", "cutleries"]:
            export_options = ["produced locally", "imported from abroad"]
        elif template["category"] in ["pets"]:
            export_options = ["raised locally", "imported from abroad"]
        else:
            export_options = ["made locally", "imported from abroad"]

        export_option = random.choice(export_options)
        return f"There is {probability}% probability that they are {export_option}."

    def _generate_probability_state_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate probability-based state distractor."""
        probability = self._get_random_probability()
        if template["category"] in ["fruits", "plants", "beverages"]:
            state_options = ["grown organically", "grown with pesticides"]
        elif template["category"] in ["books", "games", "instruments", "clothing", "desserts", "cutleries"]:
            state_options = ["produced ethically", "mass-produced"]
        elif template["category"] in ["pets"]:
            state_options = ["vaccinated", "not vaccinated yet"]
        else:
            state_options = ["high quality", "standard quality"]

        state_option = random.choice(state_options)
        return f"There is {probability}% probability that they are {state_option}."

    def _generate_weight_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate weight-based distractor."""
        weight1 = random.randint(50, 500)
        weight2 = random.randint(50, 500)
        unit = random.choice(["grams", "ounces", "pounds"])
        return f"The {specific_item1} weighs {weight1} {unit} and the {specific_item2} weighs {weight2} {unit}."

    def _generate_price_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate price-based distractor."""
        price1 = random.uniform(1.0, 50.0)
        price2 = random.uniform(1.0, 50.0)
        return f"The {specific_item1} costs ${price1:.2f} and the {specific_item2} costs ${price2:.2f}."

    def _generate_time_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate time-based distractor."""
        time1 = random.randint(1, 30)
        time2 = random.randint(1, 30)
        unit = random.choice(["minutes", "hours", "days"])
        action = random.choice(["obtained", "purchased", "received", "acquired"])
        return f"You {action} the {specific_item1} {time1} {unit} ago and the {specific_item2} {time2} {unit} ago."

    def _generate_location_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate location-based distractor."""
        locations = ["the store", "the market", "online", "from a friend", "at the mall", "downtown"]
        location1 = random.choice(locations)
        location2 = random.choice(locations)
        return f"You got the {specific_item1} from {location1} and the {specific_item2} from {location2}."

    def _generate_person_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate person-based distractor."""
        people = ["your friend", "your family", "a colleague", "a neighbor", "your teacher"]
        person1 = random.choice(people)
        person2 = random.choice(people)
        return f"The {specific_item1} was recommended by {person1} and the {specific_item2} was recommended by {person2}."

    def _generate_material_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate material-based distractor."""
        materials = ["plastic", "wood", "metal", "glass", "fabric", "ceramic"]
        material1 = random.choice(materials)
        material2 = random.choice(materials)
        return f"The {specific_item1} is made of {material1} and the {specific_item2} is made of {material2}."

    def _generate_color_distractor(self, template: dict, specific_item1: str, specific_item2: str) -> str:
        """Generate color-based distractor."""
        colors = ["red", "blue", "green", "yellow", "black", "white", "brown", "purple", "orange", "pink"]
        color1 = random.choice(colors)
        color2 = random.choice(colors)
        return f"The {specific_item1} is {color1} and the {specific_item2} is {color2}."
