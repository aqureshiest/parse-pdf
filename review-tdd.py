import requests
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

client = OpenAI()

META_PROMPT = """
Review the technical design document provided, focusing on the overall technical design, approaches, REST endpoints, and database design. Your role is to review it as a software architect.

Carefully analyze each aspect of the document:

- **Overall Technical Design and Approach**: Evaluate if the proposed design is logical and efficient. Consider scalability, maintainability, and performance.
- **REST Endpoints**: Check if the REST endpoints are well-defined, including clear input/output specifications. Assess their alignment with business requirements and standard practices.
- **Database Design**: Investigate the database schema for normalization, relationships, indexing, and overall performance. Ensure it supports the application's needs effectively.

Provide constructive feedback for improvements or confirm if it meets necessary standards and is ready for implementation.

# Output Format

- Start with a brief summary of your overall impression.
- Break down your feedback into sections: Technical Design, REST Endpoints, and Database Design.
- Offer detailed suggestions for improvement or indicate areas of success.
- Conclude with a final recommendation on whether the design is good to proceed with or needs further refinement.

# Examples

**Overall Technical Design**:  
- [Observation]: The design proposes a monolithic architecture.  
- [Feedback]: Consider adopting a microservices approach for better scalability and independent deployments.

**REST Endpoints**:  
- [Observation]: Endpoints lack standard RESTful conventions.  
- [Feedback]: Ensure endpoints use proper REST methods and naming conventions, such as GET for retrieval.

**Database Design**:  
- [Observation]: The schema includes non-normalized tables leading to data redundancy.  
- [Feedback]: Normalize tables to reduce redundancy and improve data integrity.

**Recommendation**:  
After reviewing the document, it appears [summary feedback]. It would be [decision on readiness] to proceed with certain refinements.
"""


def review_tdd(tdd: str):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": META_PROMPT,
            },
            {
                "role": "user",
                "content": "\n\n# HERE IS THE TECHNICAL DESIGN DOCUMENT TO REVIEW:\n"
                + tdd,
            },
        ],
    )

    return completion.choices[0].message.content


def main():
    path = "/Users/adeelqureshi/Downloads/erd.pdf"
    # read the pdf file in binary mode
    with open(path, "rb") as f:
        # Set the MIME type explicitly as application/pdf
        files = {"file": (path, f, "application/pdf")}
        headers = {"Accept": "application/json"}

        print("Making request")
        # make a POST request
        response = requests.post(
            "http://localhost:8000/parse-pdf", files=files, headers=headers
        )

        # output the response
        content = response.json().get("content")

        print(review_tdd(content))


main()
