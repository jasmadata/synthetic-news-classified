# Requirements: `pip install distilabel[hf-inference-endpoints]`
import os
import random
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime
from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, KeepColumns
from distilabel.steps.tasks import GenerateTextClassificationData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Difficulty(Enum):
    """Enumeration for text classification difficulty levels."""
    ELEMENTARY = "elementary"
    HIGH_SCHOOL = "high school"
    COLLEGE = "college"
    EXPERT = "expert"

class ClarityLevel(Enum):
    """Enumeration for text clarity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

@dataclass
class LLMConfig:
    """Configuration for the Language Model."""
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    base_url: str = "https://api-inference.huggingface.co/v1/"
    temperature: float = 0.8
    max_new_tokens: int = 2048
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for LLM initialization."""
        return {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }

@dataclass
class PipelineConfig:
    """Configuration for the text classification pipeline."""
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    num_generations: int = 10
    difficulty: Difficulty = Difficulty.HIGH_SCHOOL
    clarity: Optional[ClarityLevel] = None
    seed: int = field(default_factory=lambda: random.randint(0, 2**32 - 1))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    api_key: str = field(default_factory=lambda: os.getenv("HF_API_KEY", ""))

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.api_key:
            raise ValueError("HF_API_KEY environment variable must be set")
        self.output_dir.mkdir(parents=True, exist_ok=True)

class TextClassificationPipeline:
    """Handler for text classification pipeline operations."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        self.pipeline = self._create_pipeline()
        self._setup_logging()

    def _setup_logging(self):
        """Setup pipeline-specific logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.config.output_dir / f"pipeline_{timestamp}.log"
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)

    def _create_pipeline(self) -> Pipeline:
        """Create and configure the text classification pipeline."""
        logger.info("Initializing pipeline with configuration: %s", self.config)
        
        llm = InferenceEndpointsLLM(
            model_id=self.config.llm_config.model_id,
            base_url=self.config.llm_config.base_url,
            api_key=self.config.api_key,
            generation_kwargs=self.config.llm_config.to_dict(),
        )

        pipeline = Pipeline(name="textcat")
        
        # Define pipeline steps
        steps = [
            LoadDataFromDicts(data=[{"task": "text_classification"}]),
            GenerateTextClassificationData(
                llm=llm,
                seed=self.config.seed,
                difficulty=self.config.difficulty.value,
                clarity=self.config.clarity.value if self.config.clarity else None,
                num_generations=self.config.num_generations,
                output_mappings={"input_text": "text"},
            ),
            KeepColumns(columns=["text", "label"])
        ]

        # Add and connect steps
        for step in steps:
            pipeline.add_step(step)
        
        for i in range(len(steps) - 1):
            steps[i] >> steps[i + 1]

        return pipeline

    def run(self) -> Any:
        """Execute the pipeline and save results."""
        logger.info("Starting pipeline execution")
        try:
            results = self.pipeline.run()
            self._save_results(results)
            logger.info("Pipeline execution completed successfully")
            return results
        except Exception as e:
            logger.error("Pipeline execution failed: %s", str(e))
            raise

    def _save_results(self, results: Any):
        """Save pipeline results to output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.config.output_dir / f"results_{timestamp}.json"
        # Implement result saving logic here
        logger.info("Results saved to %s", output_file)

def main():
    """Main execution function."""
    try:
        config = PipelineConfig(
            llm_config=LLMConfig(),
            difficulty=Difficulty.HIGH_SCHOOL,
            clarity=ClarityLevel.MODERATE,
        )
        
        pipeline = TextClassificationPipeline(config)
        results = pipeline.run()
        return results
        
    except Exception as e:
        logger.error("Pipeline execution failed: %s", str(e))
        raise

if __name__ == "__main__":
    main()