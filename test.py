from openai import OpenAI
import os

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key = api_key)
prompt_folder = "prompt"

band_mapping = {
    "1": {"file": "1.txt", "band": "1.0 tới 3.0"},
    "2": {"file": "2.txt", "band": "3.0 tới 5.0"},
    "3": {"file": "3.txt", "band": "5.0 tới 7.0"},
    "4": {"file": "4.txt", "band": "7.0 tới 9.0"}
}

print( "select band : ")

for key, value  in band_mapping.items():
    print(f"{key}: band { value['band']}")

choice = input("Nhap so tu 1 toi 4 de lua chon band :").strip()

if choice not in band_mapping:
    print("lua chon khong hop lee")
else:
    selected_file = band_mapping[choice]['file']
    band = band_mapping[choice]["band"]
    file_path = os.path.join(prompt_folder, selected_file)

    with open(file_path, 'r', encoding='utf-8') as f:
        prompt = f.read()

    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [{"role" : "user",
                     "content" : prompt},
        ]
    )
    bai_tap = response.choices[0].message.content.strip()

    output_folder = "output"
    os.makedirs(output_folder,exist_ok=True)
    output_file = os.path.join(output_folder, f"output_{selected_file}")
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(bai_tap)

    print("done")