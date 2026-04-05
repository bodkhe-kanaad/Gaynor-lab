import json
 
path = '/Users/bodkhe.kanaad/Gaynor Lab/all_blank_crops/md_results.json'
data = json.load(open(path))
 
print(f"Top-level type : {type(data)}")
print(f"Top-level keys : {list(data.keys()) if isinstance(data, dict) else 'n/a'}")
print()
 
if isinstance(data, dict):
    if "images" in data:
        images = data["images"]
        print(f"Standard MD format. Total images: {len(images)}")
        for img in images:
            dets = img.get("detections") or []
            print(f"  {img['file']}  -> {len(dets)} detections")
            for d in dets:
                print(f"    conf={d['conf']:.4f}  category={d['category']}")
    else:
        print(f"Our format. Total image paths: {len(data)}")
        for path_key, dets in data.items():
            print(f"  {path_key}  -> {len(dets)} detections")
            for d in dets:
                print(f"    conf={d['conf']:.4f}  category={d['category']}")
else:
    print("Unexpected format:")
    print(json.dumps(data, indent=2)[:2000])