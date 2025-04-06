import os
import logging
import datetime
import io
import base64
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Embedded API configuration
OPENAI_API_KEY = ""  # Replace with your actual API key
OPENAI_BASE_URL = "" #add your actual base url


class EnhancedLLMAgent:
    """An enhanced agent with PDF generation and visualization capabilities."""
    
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
        
        logger.info("EnhancedLLMAgent initialized successfully")
    
    def execute(self, instruction: str) -> Dict[str, Any]:
        """Execute the given instruction using the LLM."""
        logger.info(f"Executing instruction: {instruction}")
        
        try:
            # Check if visualization is requested
            needs_visualization = any(keyword in instruction.lower() for keyword in 
                                     ["chart", "graph", "plot", "figure", "visualization", "visualize", "diagram"])
            
            # Check if PDF is requested
            needs_pdf = any(keyword in instruction.lower() for keyword in 
                           ["pdf", "document", "report", "export"])
            
            # Create a prompt that instructs the LLM to handle the task
            prompt = f"""
            I need you to help me with the following task:
            
            {instruction}
            
            Please approach this task step by step:
            1. Think about what information is needed
            2. Consider what you already know about this topic
            3. Provide a detailed response that addresses the task
            """
            
            # Add visualization instructions if needed
            if needs_visualization:
                prompt += """
                4. Include data for visualizations in the following JSON format:
                
                ```json
                {{
                    "charts": [
                        {{
                            "title": "Chart Title",
                            "type": "line|bar|pie|scatter",
                            "x_label": "X-Axis Label",
                            "y_label": "Y-Axis Label",
                            "data": {{
                                "labels": ["Label1", "Label2", "Label3", ...],
                                "datasets": [
                                    {{
                                        "label": "Dataset Label",
                                        "values": [value1, value2, value3, ...]
                                    }},
                                    ...
                                ]
                            }}
                        }},
                        ...
                    ]
                }}
                ```
                
                Provide realistic and representative data based on your knowledge of the topic.
                """
            
            # Get response from the LLM
            response = self.llm.invoke(prompt)
            content = response.content
            
            # Process the response
            result = {
                "text_content": content,
                "charts_data": None,
                "pdf_path": None
            }
            
            # Extract chart data if present
            if needs_visualization:
                try:
                    # Try to extract JSON data for charts
                    import json
                    import re
                    
                    # Look for JSON blocks in the content
                    json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        charts_data = json.loads(json_str)
                        result["charts_data"] = charts_data
                        
                        # Create visualizations
                        chart_paths = self._create_visualizations(charts_data)
                        result["chart_paths"] = chart_paths
                    else:
                        # If no JSON block found, try to extract data in a different way
                        logger.warning("No JSON chart data found in the response")
                except Exception as e:
                    logger.error(f"Error extracting chart data: {str(e)}")
            
            # Generate PDF if requested
            if needs_pdf:
                try:
                    pdf_path = self._generate_pdf(
                        content, 
                        result.get("charts_data"), 
                        f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    )
                    result["pdf_path"] = pdf_path
                except Exception as e:
                    logger.error(f"Error generating PDF: {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing instruction: {str(e)}")
            return {"text_content": f"Error: {str(e)}", "charts_data": None, "pdf_path": None}
    
    def _create_visualizations(self, charts_data: Dict[str, Any]) -> List[str]:
        """Create visualizations based on the provided data."""
        chart_paths = []
        
        try:
            # Process each chart in the data
            for i, chart in enumerate(charts_data.get("charts", [])):
                plt.figure(figsize=(10, 6))
                
                chart_type = chart.get("type", "bar")
                title = chart.get("title", f"Chart {i+1}")
                x_label = chart.get("x_label", "")
                y_label = chart.get("y_label", "")
                
                data = chart.get("data", {})
                labels = data.get("labels", [])
                datasets = data.get("datasets", [])
                
                if chart_type == "bar":
                    for j, dataset in enumerate(datasets):
                        dataset_label = dataset.get("label", f"Dataset {j+1}")
                        values = dataset.get("values", [])
                        x = np.arange(len(labels))
                        width = 0.8 / len(datasets)
                        offset = j * width - (len(datasets) - 1) * width / 2
                        plt.bar(x + offset, values, width, label=dataset_label)
                    plt.xticks(np.arange(len(labels)), labels)
                
                elif chart_type == "line":
                    for j, dataset in enumerate(datasets):
                        dataset_label = dataset.get("label", f"Dataset {j+1}")
                        values = dataset.get("values", [])
                        plt.plot(labels, values, marker='o', label=dataset_label)
                
                elif chart_type == "pie":
                    # Use only the first dataset for pie charts
                    if datasets:
                        values = datasets[0].get("values", [])
                        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                
                elif chart_type == "scatter":
                    for j, dataset in enumerate(datasets):
                        dataset_label = dataset.get("label", f"Dataset {j+1}")
                        values = dataset.get("values", [])
                        # For scatter plots, we need x and y values
                        if len(labels) == len(values):
                            plt.scatter(labels, values, label=dataset_label)
                
                plt.title(title)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                if len(datasets) > 1:
                    plt.legend()
                plt.tight_layout()
                
                # Save the chart
                chart_filename = f"chart_{i+1}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(chart_filename)
                plt.close()
                
                chart_paths.append(chart_filename)
        
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
        
        return chart_paths
    
    def _generate_pdf(self, content: str, charts_data: Optional[Dict[str, Any]], filename: str) -> str:
        """Generate a PDF report with the content and charts."""
        try:
            # Create a PDF document
            doc = SimpleDocTemplate(filename, pagesize=letter)
            styles = getSampleStyleSheet()
            
            # Create custom styles
            title_style = ParagraphStyle(
                'TitleStyle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=12
            )
            
            heading_style = ParagraphStyle(
                'HeadingStyle',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=10
            )
            
            normal_style = ParagraphStyle(
                'NormalStyle',
                parent=styles['Normal'],
                fontSize=10,
                spaceAfter=8
            )
            
            # Process content for PDF
            story = []
            
            # Extract title from content (assuming first line is title)
            lines = content.split('\n')
            title = "Report"
            content_start = 0
            
            for i, line in enumerate(lines):
                if line.strip().startswith('# '):
                    title = line.strip('# ')
                    content_start = i + 1
                    break
            
            # Add title
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 0.25*inch))
            
            # Process content by sections
            current_section = []
            in_code_block = False
            
            for line in lines[content_start:]:
                # Handle code blocks
                if line.strip().startswith('```'):
                    in_code_block = not in_code_block
                    continue
                
                # Handle headings
                if not in_code_block and line.strip().startswith('## '):
                    # Add previous section if exists
                    if current_section:
                        story.append(Paragraph('\n'.join(current_section), normal_style))
                        current_section = []
                    
                    # Add new heading
                    heading = line.strip('## ')
                    story.append(Spacer(1, 0.2*inch))
                    story.append(Paragraph(heading, heading_style))
                    continue
                
                # Skip JSON blocks
                if in_code_block and 'json' in line:
                    continue
                
                # Add line to current section
                if not (in_code_block and line.strip().startswith('{')):
                    current_section.append(line)
            
            # Add final section
            if current_section:
                story.append(Paragraph('\n'.join(current_section), normal_style))
            
            # Add charts if available
            if charts_data and hasattr(self, '_create_visualizations'):
                chart_paths = self._create_visualizations(charts_data)
                
                if chart_paths:
                    story.append(Spacer(1, 0.3*inch))
                    story.append(Paragraph("Visualizations", heading_style))
                    
                    for chart_path in chart_paths:
                        story.append(Spacer(1, 0.2*inch))
                        img = Image(chart_path, width=6*inch, height=3.5*inch)
                        story.append(img)
            
            # Build the PDF
            doc.build(story)
            logger.info(f"PDF report generated: {filename}")
            
            return filename
            
        except Exception as e:
            logger.error(f"Error generating PDF: {str(e)}")
            return ""


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
    print("ENHANCED AI ASSISTANT".center(50))
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
    print("\n1. Basic: Research renewable energy trends with charts")
    print("2. Intermediate: Compare global economic indicators with visualizations")
    print("3. Advanced: Analyze climate change data and create a comprehensive PDF report")
    
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
    print("- Research renewable energy trends and create a PDF report with charts")
    print("- Analyze global smartphone market share and visualize the data")
    
    instruction = input("\nYour instruction: ")
    return instruction


def run_predefined_task(agent, task_name):
    """Run a predefined task."""
    tasks = {
        "basic": "Research renewable energy trends over the past decade. Create charts showing the growth "
                "of solar, wind, and hydroelectric power. Generate a PDF report with your findings and visualizations.",
        
        "intermediate": "Compare key economic indicators (GDP growth, inflation, unemployment) for the top 5 "
                       "global economies over the past 5 years. Create appropriate visualizations for each indicator "
                       "and compile your analysis into a comprehensive PDF report.",
        
        "advanced": "Analyze climate change data including global temperature changes, sea level rise, and carbon "
                   "emissions over the past century. Create multiple visualizations showing these trends and their "
                   "correlations. Compile a detailed PDF report with your analysis, visualizations, and potential "
                   "future scenarios based on current trends."
    }
    
    if task_name not in tasks:
        logger.error(f"Unknown task: {task_name}")
        return f"Error: Unknown task '{task_name}'"
    
    instruction = tasks[task_name]
    print(f"\nRunning task: {task_name}")
    print(f"Instruction: {instruction}")
    print("\nProcessing... (this may take a few minutes)")
    
    # Execute the instruction using the agent
    result = agent.execute(instruction)
    
    # Save the text content to a file
    text_content = result["text_content"]
    output_file = save_output_to_file(text_content, f"task_{task_name}")
    
    # Return information about all outputs
    output_info = f"Task: {task_name}\n"
    output_info += f"Text output saved to: {output_file}\n"
    
    if result.get("chart_paths"):
        output_info += f"Charts saved as: {', '.join(result['chart_paths'])}\n"
    
    if result.get("pdf_path"):
        output_info += f"PDF report saved as: {result['pdf_path']}\n"
    
    return output_info


def run_custom_instruction(agent, instruction):
    """Run a custom instruction."""
    print(f"\nRunning your custom instruction...")
    print("\nProcessing... (this may take a few minutes)")
    
    # Execute the instruction using the agent
    result = agent.execute(instruction)
    
    # Save the text content to a file
    text_content = result["text_content"]
    output_file = save_output_to_file(text_content, "custom_instruction")
    
    # Return information about all outputs
    output_info = f"Custom Instruction: {instruction}\n"
    output_info += f"Text output saved to: {output_file}\n"
    
    if result.get("chart_paths"):
        output_info += f"Charts saved as: {', '.join(result['chart_paths'])}\n"
    
    if result.get("pdf_path"):
        output_info += f"PDF report saved as: {result['pdf_path']}\n"
    
    return output_info


def main():
    """Main function to run the interactive AI assistant."""
    print("\nInitializing Enhanced AI Assistant...")
    
    try:
        # Initialize the agent with embedded API key
        agent = EnhancedLLMAgent(
            openai_api_key=OPENAI_API_KEY,
            openai_base_url=OPENAI_BASE_URL
        )
        
        while True:
            choice = display_menu()
            
            if choice == 1:  # Run predefined task
                task = select_predefined_task()
                output_info = run_predefined_task(agent, task)
                print(f"\nTask completed successfully!")
                print(f"\n{output_info}")
                
            elif choice == 2:  # Run custom instruction
                instruction = get_custom_instruction()
                output_info = run_custom_instruction(agent, instruction)
                print(f"\nTask completed successfully!")
                print(f"\n{output_info}")
                
            else:  # Exit
                print("\nThank you for using the Enhanced AI Assistant. Goodbye!")
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
