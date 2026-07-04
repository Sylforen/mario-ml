import yaml
from pathlib import Path
from tqdm import tqdm


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_sprite_paths(node):
    sprites = []

    if isinstance(node, dict):
        for k, v in node.items():
            if k == "sprites" and isinstance(v, list):
                sprites.extend(v)
            else:
                sprites.extend(extract_sprite_paths(v))

    elif isinstance(node, list):
        for item in node:
            sprites.extend(extract_sprite_paths(item))

    return sprites


def find_missing_sprites(yaml_data, base_dir: Path):
    sprites = extract_sprite_paths(yaml_data)

    missing = []
    existing = []

    for sprite_path in tqdm(sprites, desc="Checking sprites"):
        full_path = base_dir / sprite_path

        if full_path.exists():
            existing.append(sprite_path)
        else:
            missing.append(sprite_path)

    return existing, missing


def main():
    from pathlib import Path

    SCRIPT_DIR = Path(__file__).resolve().parent
    yaml_path = SCRIPT_DIR / "../sprite_registry.yaml"
    yaml_path = yaml_path.resolve()
    base_dir = SCRIPT_DIR.parent  # feature-matching/
    data = load_yaml(yaml_path)

    existing, missing = find_missing_sprites(data, base_dir)

    print("\n=== SPRITE CHECK REPORT ===\n")
    print(f"Total sprites: {len(existing) + len(missing)}")
    print(f"Existing: {len(existing)}")
    print(f"Missing: {len(missing)}\n")

    if missing:
        print("Missing files:")
        for m in missing:
            print(f"  - {m}")


if __name__ == "__main__":
    main()