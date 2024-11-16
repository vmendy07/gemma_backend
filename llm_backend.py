from openai import OpenAI
from typing import Dict
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Enable CORS for all domains for testing. In production, specify your frontend domain
CORS(app)

class QuizAnalyzer:
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
    def analyze_quiz_data(self, quiz_data: Dict) -> str:
        try:
            # Format the data for analysis
            prompt = f"""
Analyze this coding quiz platform data and provide insights back to the user. The data is structured as follows:

{json.dumps(quiz_data, indent=2)}

Please provide:
1. Key insights about the user's performance patterns
2. Recommendations for improving coding quiz effectiveness
3. Identification of coding knowledge gaps
4. Suggestions for ways to improve for next time
"""
            
            # Get insights using the LLM
            completion = self.client.chat.completions.create(
                model="google/gemini-pro",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing coding quiz data. Provide concise, actionable insights about user performance, difficulty levels, and learning patterns."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error analyzing quiz data: {str(e)}")

# Input validation helper
def validate_quiz_data(data: Dict) -> bool:
    required_fields = ['platform_name', 'quizzes', 'platform_metadata']
    return all(field in data for field in required_fields)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Service is running"
    }), 200

@app.route('/analyze-quiz', methods=['POST'])
def analyze_quiz():
    try:
        # Get quiz data from request
        quiz_data = request.json
        
        # Validate input
        if not quiz_data:
            return jsonify({
                "status": "error",
                "message": "No data provided"
            }), 400
            
        if not validate_quiz_data(quiz_data):
            return jsonify({
                "status": "error",
                "message": "Invalid quiz data format"
            }), 400
        
        # Initialize analyzer with API key from environment variable
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            return jsonify({
                "status": "error",
                "message": "API key not configured"
            }), 500
            
        analyzer = QuizAnalyzer(api_key)
        
        # Get analysis
        insights = analyzer.analyze_quiz_data(quiz_data)
        
        return jsonify({
            "status": "success",
            "insights": insights
        }), 200
        
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Internal server error"
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "status": "error",
        "message": "Resource not found"
    }), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        "status": "error",
        "message": "Method not allowed"
    }), 405

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "status": "error",
        "message": "Internal server error"
    }), 500

if __name__ == "__main__":
    # Get port from environment variable or default to 5000
    port = int(os.getenv('PORT', 5000))
    
    # In production, you would typically use gunicorn instead
    app.run(
        host='0.0.0.0',  # Makes the server externally visible
        port=port,
        debug=os.getenv('FLASK_ENV') == 'development'  # Only enable debug in development
    )