# Gloss Retrieval System

Hệ thống retrieval glosses với SMPLX + Embeddings.

## Database Structure

```python
{
    'GLOSS_NAME': {
        'embedding': np.array([...]),      # ProtonX embedding
        'smplx': [frame1, frame2, ...] or None  # SMPLX frames
    }
}
```

---

## Setup

### 1. Build Database (First time)

```bash
# Build database với batch processing
python build_gloss_db.py

# Tùy chọn batch size
python build_gloss_db.py --batch_size 50

# Start fresh (ignore checkpoint)
python build_gloss_db.py --no_resume
```

**Features:**
- Batch processing (100 glosses/batch)
- Auto checkpoint after each batch
- Resume from checkpoint on error
- Retry failed requests
- Progress tracking

### 2. Retry Failed Glosses

```bash
# Nếu có glosses failed, retry sau
python retry_failed.py
```

### 3. Test

```bash
python test.py
```

---

## Build Database Workflow

```
1. Load gloss_dictionary.pkl (3251 glosses)
2. Load smplx_all.pkl (8 glosses with SMPLX)
3. Process in batches:
   - Batch 1: Glosses 1-100
     → Compute embeddings
     → Save checkpoint
   - Batch 2: Glosses 101-200
     → Compute embeddings
     → Save checkpoint
   ...
4. Save final database
5. Remove checkpoint
```

### Resume on Error

Nếu script bị gián đoạn (timeout, disconnect, Ctrl+C):

```bash
# Chạy lại - tự động resume từ checkpoint
python build_gloss_db.py
```

Script sẽ:
- Load checkpoint
- Skip glosses đã xử lý
- Tiếp tục từ batch tiếp theo

---

## Progress Tracking

```
============================================================
BUILD GLOSS DATABASE (Batch Mode)
============================================================

[1/3] Loading glosses...
  3251 glosses

[2/3] Loading SMPLX data...
  8 glosses with SMPLX
    AN_UONG: 58 frames
    BO_ME: 46 frames
    ...

[3/3] Computing embeddings for 3251 glosses...
  Batch size: 100
  Total batches: 33

  Batch 1/33 (100 glosses)
  Processing batch 1: 100%|| 100/100 [2:15<00:00]
   Checkpoint saved: data/gloss_db_checkpoint.pkl
  Progress: 100/3251 glosses

  Batch 2/33 (100 glosses)
  Processing batch 2: 100%|| 100/100 [2:18<00:00]
   Checkpoint saved
  Progress: 200/3251 glosses

  ...

  Batch 33/33 (51 glosses)
  Processing batch 33: 100%|| 51/51 [1:12<00:00]
   Checkpoint saved
  Progress: 3251/3251 glosses

  Saving final database to data/gloss_db.pkl...
   Removed checkpoint file

============================================================
SUMMARY
============================================================
Total glosses: 3251
Processed: 3251
Failed: 0
With SMPLX: 8

Output: data/gloss_db.pkl
Size: 14.52 MB
```

---

## Error Handling

### API Errors (503, Timeout)

Script tự động:
1. Retry 3 lần với delay 5s
2. Skip nếu vẫn fail
3. Continue với glosses khác
4. Save checkpoint sau mỗi batch

### Retry Failed Glosses

```bash
# Check missing glosses
python3 -c "
import pickle
with open('data/gloss_dictionary.pkl', 'rb') as f:
    all_glosses = set(pickle.load(f))
with open('data/gloss_db.pkl', 'rb') as f:
    db_glosses = set(pickle.load(f).keys())
missing = all_glosses - db_glosses
print(f'Missing: {len(missing)} glosses')
"

# Retry
python retry_failed.py
```

---

## Files

```
text2sign_pipeline/
 data/
    gloss_dictionary.pkl           # Input: 3251 glosses
    gloss_db.pkl                   # Output: DB with embeddings
    gloss_db_checkpoint.pkl        # Checkpoint (auto-removed)

 models/data/
    smplx_all.pkl                  # SMPLX data (8 glosses)

 build_gloss_db.py                  # Build database (batch mode)
 retry_failed.py                    # Retry failed glosses
 test.py                            # Test retrieval
 README.md
```

---

## Tips

### Batch Size

- **Default: 100** - Good balance
- **Smaller (50)**: More checkpoints, slower
- **Larger (200)**: Fewer checkpoints, risk more loss on error

```bash
# Small batches for unreliable connection
python build_gloss_db.py --batch_size 50

# Large batches for stable connection
python build_gloss_db.py --batch_size 200
```

### Resume

```bash
# Resume from checkpoint (default)
python build_gloss_db.py

# Start fresh (delete checkpoint)
python build_gloss_db.py --no_resume
```

### Monitor Progress

```bash
# While running, check in another terminal
python3 -c "
import pickle
with open('data/gloss_db_checkpoint.pkl', 'rb') as f:
    db = pickle.load(f)
print(f'Progress: {len(db)} glosses')
"
```

---

## Verify Database

```bash
python3 -c "
import pickle

# Load database
with open('data/gloss_db.pkl', 'rb') as f:
    db = pickle.load(f)

print(f'Total glosses: {len(db)}')
print(f'With SMPLX: {sum(1 for v in db.values() if v[\"smplx\"])}')
print()

# Sample
sample = list(db.items())[0]
name, data = sample
print(f'Sample: {name}')
print(f'  Embedding: {data[\"embedding\"].shape}')
print(f'  SMPLX: {len(data[\"smplx\"]) if data[\"smplx\"] else None} frames')
"
```

---

## Integration

### Use in retrieval

```python
from sign_retrieval import GlossRetriever, RetrievalConfig

config = RetrievalConfig(
    protonx_api_key='your_key',
    embedding_threshold=0.95
)
retriever = GlossRetriever(config)

result = retriever.retrieve("Chúng tôi ăn uống")
print(result['output'])  # CHÚNG_TÔI ĂN_UỐNG

# Access SMPLX
for g in result['glosses']:
    if g['smplx']:
        print(f"{g['name']}: {g['num_frames']} frames")
        # Render with g['smplx']
```

---

## Requirements

- Python 3.8+
- ProtonX API key
- ~15 MB disk space for database
- Internet connection for API calls

---

## ⏱ Performance

- **Time**: ~1.5s per gloss (with API)
- **Total**: ~80 minutes for 3251 glosses
- **Batch**: Save checkpoint every ~2-3 minutes
- **Resume**: Instant (skip processed glosses)
