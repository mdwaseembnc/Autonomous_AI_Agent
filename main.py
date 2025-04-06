import os
import logging
import datetime
from typing import Optional
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Embedded API configuration
OPENAI_API_KEY = ""  # Replace with your actual API key
OPENAI_BASE_URL = "" #enter your_base_url


class SimpleLLMAgent:
    """A simplified agent that uses only the LLM capabilities."""
    
    def __init__(self, openai_api_key: str, openai_base_url: Optional[str] = None):
        """Initialize the agent with the LLM."""
        self.api_key = openai_api_key
        self.base_url = openai_base_url
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        logger.info("SimpleLLMAgent initialized successfully")
    
    def execute(self, instruction: str) -> str:
        """Execute the given instruction using the LLM."""
        logger.info(f"Executing instruction: {instruction}")
        
        try:
            # Create a prompt that instructs the LLM to handle the task
            prompt = f"""
            I need you to help me with the following task:
            
            {instruction}
            
            Please approach this task step by step:
            1. Think about what information is needed
            2. Consider what you already know about this topic
            3. Provide a detailed response that addresses the task
            4. If you're asked to create a file or document, describe what would be in it
            
            Note: You don't have access to the internet or external tools, so use your knowledge as of your training data.
            If the task requires current information that you don't have, please acknowledge this limitation.
            """
            
            # Get response from the LLM
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            logger.error(f"Error executing instruction: {str(e)}")
            return f"Error: {str(e)}"


def save_output_to_file(content, prefix="output"):
    """Save the output content to a text file with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)
    
    return filename


def display_menu():
    """Display the main menu options."""
    print("\n" + "="*50)
    print("AI ASSISTANT".center(50))
    print("="*50)
    print("\nWhat would you like to do?")
    print("1. Run a predefined task")
    print("2. Enter a custom instruction")
    print("3. Exit")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-3): "))
            if 1 <= choice <= 3:
                return choice
            else:
                print("Please enter a number between 1 and 3.")
        except ValueError:
            print("Please enter a valid number.")


def select_predefined_task():
    """Let the user select a predefined task."""
    print("\n" + "-"*50)
    print("PREDEFINED TASKS".center(50))
    print("-"*50)
    print("\n1. Basic: Find information about the latest AI developments")
    print("2. Intermediate: Compare latest iPhone and Samsung smartphone features")
    print("3. Advanced: Explain current trends in renewable energy technology")
    
    tasks = {
        1: "basic",
        2: "intermediate",
        3: "advanced"
    }
    
    while True:
        try:
            choice = int(input("\nSelect a task (1-3): "))
            if 1 <= choice <= 3:
                return tasks[choice]
            else:
                print("Please enter a number between 1 and 3.")
        except ValueError:
            print("Please enter a valid number.")


def get_custom_instruction():
    """Get a custom instruction from the user."""
    print("\n" + "-"*50)
    print("CUSTOM INSTRUCTION".center(50))
    print("-"*50)
    print("\nEnter your instruction for the AI assistant.")
    print("Examples:")
    print("- Explain the health benefits of meditation")
    print("- Describe the top tourist attractions in Tokyo")
    
    instruction = input("\nYour instruction: ")
    return instruction


def run_predefined_task(agent, task_name):
    """Run a predefined task."""
    tasks = {
        "basic": "Provide information about the latest developments in artificial intelligence. "
                "Focus on recent advancements in machine learning, natural language processing, "
                "and computer vision. Include potential applications and impacts.",
        
        "intermediate": "Compare the latest iPhone and Samsung flagship smartphones. "
                       "Include details about their processors, cameras, displays, battery life, "
                       "and unique features. Create a balanced comparison highlighting the strengths "
                       "and weaknesses of each device.",
        
        "advanced": "Explain the current trends in renewable energy technology. Discuss the recent "
                   "developments in solar, wind, and hydroelectric power. Include information about "
                   "efficiency improvements, cost reductions, and adoption rates. Also mention emerging "
                   "technologies and future prospects in this field."
    }
    
    if task_name not in tasks:
        logger.error(f"Unknown task: {task_name}")
        return f"Error: Unknown task '{task_name}'"
    
    instruction = tasks[task_name]
    print(f"\nRunning task: {task_name}")
    print(f"Instruction: {instruction}")
    print("\nProcessing... (this may take a few moments)")
    
    # Execute the instruction using the agent
    result = agent.execute(instruction)
    
    # Save the result to a text file
    output_content = f"Task: {task_name}\n"
    output_content += f"Instruction: {instruction}\n\n"
    output_content += f"Result:\n{result}\n"
    
    output_file = save_output_to_file(output_content, f"task_{task_name}")
    
    return output_file


def run_custom_instruction(agent, instruction):
    """Run a custom instruction."""
    print(f"\nRunning your custom instruction...")
    print("\nProcessing... (this may take a few moments)")
    
    # Execute the instruction using the agent
    result = agent.execute(instruction)
    
    # Save the result to a text file
    output_content = f"Custom Instruction: {instruction}\n\n"
    output_content += f"Result:\n{result}\n"
    
    output_file = save_output_to_file(output_content, "custom_instruction")
    
    return output_file


def main():
    """Main function to run the interactive AI assistant."""
    print("\nInitializing AI Assistant...")
    
    try:
        # Initialize the agent with embedded API key
        agent = SimpleLLMAgent(
            openai_api_key=OPENAI_API_KEY,
            openai_base_url=OPENAI_BASE_URL
        )
        
        while True:
            choice = display_menu()
            
            if choice == 1:  # Run predefined task
                task = select_predefined_task()
                output_file = run_predefined_task(agent, task)
                print(f"\nTask completed successfully!")
                print(f"Results saved to: {output_file}")
                
            elif choice == 2:  # Run custom instruction
                instruction = get_custom_instruction()
                output_file = run_custom_instruction(agent, instruction)
                print(f"\nTask completed successfully!")
                print(f"Results saved to: {output_file}")
                
            else:  # Exit
                print("\nThank you for using the AI Assistant. Goodbye!")
                break
                
            input("\nPress Enter to continue...")
            
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        
        # Save error information to a file
        error_content = f"Error occurred at {datetime.datetime.now()}\n"
        error_content += f"Error message: {str(e)}\n"
        error_file = save_output_to_file(error_content, "error")
        
        print(f"\nExecution failed: {str(e)}")
        print(f"Error details saved to: {error_file}")


if __name__ == "__main__":
    main()
