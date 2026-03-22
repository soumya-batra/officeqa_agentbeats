import os
from openai import OpenAI

SYSTEM_PROMPT = """You are a helpful agent that answers questions about the U.S. Treasury Bulletin. Ensure numerical accuracy and full precision in calculations while answering the question.

Provide your final answer in the following required format:
<REASONING>
[steps and calculations]
</REASONING>
<FINAL_ANSWER>
[value]
</FINAL_ANSWER>

If you do not produce a <FINAL_ANSWER> tag with the canonical final answer enclosed, your response will be considered incorrect.

"""

os.environ["NEBIUS_API_KEY"] = "v1.CmMKHHN0YXRpY2tleS1lMDBrY2oxdnhhYmY1eWE0ZHcSIXNlcnZpY2VhY2NvdW50LWUwMHpiMGFkOHdkZWY2bjJ2ZTIMCOiC0M0GEPD2zvcCOgsI6IXomAcQgLblf0ACWgNlMDA.AAAAAAAAAAG568ffA7lcylbFMKdDDs242cz4WV04CBuoNmAXjq7F5loTl12-Y4jqvUqhH201Bci9cGRrpCkkcVThLOQ6L3EA"
client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)

completion = client.chat.completions.create(
    model="nvidia/nemotron-3-super-120b-a12b",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": """What were the total expenditures (in millions of nominal dollars) for U.S national defense in the calendar year of 1940?"""
        }
    ],
    temperature=0
)

#print(completion.to_json())
print(completion.choices[0].message.content)