from typesafe_sdk import Choice, Noul, TypeSafeClient
import sys


text = sys.stdin.read()


with TypeSafeClient() as client:
    response = client.system_one(
        state={
            "content": text
        },

        questions={
            "security_sensitive": Noul(
                instructions=(
                    "Does this code change appear security-sensitive?"
                )
            ),

            "risk": Choice(
                instructions="What is the overall implementation risk?",
                criteria={
                    "low": "Small isolated change with limited failure impact.",
                    "medium": "Meaningful behavior change or moderate regression risk.",
                    "high": "Security, data integrity, concurrency, authentication, or major system behavior could be affected.",
                },
            ),

            "tests_needed": Noul(
                instructions=(
                    "Does this change require additional tests?"
                )
            ),
        },
    )


security = response.answers["security_sensitive"].noul
tests = response.answers["tests_needed"].noul
risk = response.choices["risk"]

print(f"risk: {risk.choice}")
print(f"security_sensitive_probability: {security:.3f}")
print(f"tests_needed_probability: {tests:.3f}")
