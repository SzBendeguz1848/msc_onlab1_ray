# llm-d-inference-sim — rövid jegyzet

Források:

- [llm-d/llm-d-inference-sim](https://github.com/llm-d/llm-d-inference-sim)
- [configuration.md](https://github.com/llm-d/llm-d-inference-sim/blob/main/docs/configuration.md)
- [kv-cache.md](https://github.com/llm-d/llm-d-inference-sim/blob/main/docs/kv-cache.md)
- [latency-simulation.md](https://github.com/llm-d/llm-d-inference-sim/blob/main/docs/latency-simulation.md)

## Mi ez?

OpenAI-kompatibilis **vLLM-szimulátor GPU és valódi modell nélkül**. Routing, scheduling, latency, KV/prefix cache és metrikák tesztelésére használjuk.

- Válasz: tipikusan `POST /v1/chat/completions` → `chat.completion` JSON.
- Indítás: CLI flag-ek vagy YAML (`--config`). Precedencia: **CLI > env > YAML > default**.
- Startup flag változtatás = **processz / konténer újraindítás**.
- HTTP kérés mezői (`messages`, `max_tokens`, `stream`, …) futás közben változtathatók.

---

## Alap flag-ek

| Flag | Default / megjegyzés |
|------|----------------------|
| `--model` | **Kötelező.** Nem létező név (pl. `fake-model`) → gyors simulated tokenizer. Valós HF név → pontosabb tokenizálás; KV-hez render sidecar kell. |
| `--port` | `8000` |
| `--mode` | `random` (szintetikus szöveg) vagy `echo` (visszaadja a last user message-t; `max_tokens` echo-nál nem számít) |
| `--max-model-len` | kontextusablak (default `1024`) |
| `--max-num-seqs` | egyszerre futó kérések max (default `5`) |
| `--max-waiting-queue-length` | várakozási sor max (default `1000`) |

**API (röviden):** `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics`. Swagger UI nincs; a végpontok a repo [api.md](https://github.com/llm-d/llm-d-inference-sim/blob/main/docs/api.md) doksijában vannak.

---

## Példa: alap YAML config

CLI helyett (vagy mellett) `--config` fájllal töltjük be. A latency mezőket `latencies:` alá tesszük.

```yaml
    engine: vllm
    port: 8000
    model: "fake-model"
    mode: random

    max-model-len: 1024
    max-num-seqs: 5
    max-waiting-queue-length: 1000

    latency-calculator: constant
    latencies:
      time-to-first-token: 100ms
      inter-token-latency: 20ms
      time-factor-under-load: 1.0
```

Időtartamok szöveges egységgel: szám + `ms` / `s` / `us` (pl. `100ms`, `1.5s`). Egész szám másodpercben már elavult a doksi szerint.

---

## Latency és TTFT

Egy kérés teljes eltelt ideje (indulástól a válasz végéig) nagyjából:

```text
összidő ≈ prefill_idő + decode_idő
```

- **Prefill** → ezt mérjük **TTFT**-ként (első tokenig eltelt idő).
- **Decode** → a többi token; `(output_tokens − 1) × inter-token-latency`.

A latency config **közvetlenül hat a TTFT-re** (és az E2E-re).

### Calculator: hogyan számolódik a TTFT?

`--latency-calculator`:

| Érték | TTFT (helyi / aggregated prefill) |
|-------|-----------------------------------|
| `constant` | fix: `time-to-first-token` — **független** a prompt hosszától és a cache hittől |
| `per-token` | `prefill-overhead + (prompt_tokens − cached_tokens) × prefill-time-per-token` |
| üres / legacy | ha `time-to-first-token` ≠ 0 → constant-szerű; különben per-token felbontás |

**Fontos:** ha be van állítva nem-nulla `--time-to-first-token`, a flat TTFT győz. Ilyenkor a prefix cache hit **nem rövidíti** a TTFT-et. Cache-érzékeny TTFT-hez `latency-calculator: per-token`, `prefill-time-per-token` (és overhead) kell, és **ne** legyen aktív flat TTFT.

### Fő latency flag-ek

| Flag | Szerep |
|------|--------|
| `--time-to-first-token` | fix TTFT (`constant` / legacy) |
| `--time-to-first-token-std-dev` | jitter a TTFT körül (max ~30% a mean-ből) |
| `--inter-token-latency` | idő két kimeneti token között (ITL / TPOT) |
| `--inter-token-latency-std-dev` | ITL jitter |
| `--prefill-overhead` | `per-token`: fix rész a prefillben |
| `--prefill-time-per-token` | `per-token`: költség prompt-tokenenként (csak nem cache-elt) |
| `--prefill-time-std-dev` | jitter a számolt prefill összesen |
| `--time-factor-under-load` | default `1.0`; ha >1 és tele a concurrency, a GPU-bound idők (TTFT/ITL/prefill) szorzódnak |

P/D (diszaggregált) esetén a TTFT helyett / mellett a **KV transfer** latency számít (`kv-cache-transfer-latency` vagy per-token transfer) — ez más útvonal, mint a helyi prefix cache.

---

## KV cache / prefix cache

A sim **prefix-cache alrendszere** a vLLM KV viselkedését utánozza GPU nélkül: token-blokkokat tart számon, `cached_tokens`-t ír a válaszba, és (opcionálisan) ZMQ-n közli a külvilággal, melyik „pod” milyen blokkokat tart. Részletek: [kv-cache.md](https://github.com/llm-d/llm-d-inference-sim/blob/main/docs/kv-cache.md).

### Bekapcsolás

```bash
--enable-kvcache
```

**Kötelező mellé:**

1. **`POD_IP` env** — nélküle KV módban nem indul.
2. **vLLM render sidecar** + **valós HF `--model`** + `--render-url` — blokk-hashekhez valódi token ID kell.

Kapcsolódó: `--kv-cache-size`, `--block-size` (8/16/32/64/128), `--hash-seed`, `--zmq-endpoint` (default `tcp://127.0.0.1:5557`).

A sim **`--model` is a HF név** (ugyanaz, mint a renderé) + `--render-url` — nem marad `fake-model`. A render tokenizál; a sim továbbra is mock választ ad.

### Mi az a sidecar (render)?

**Sidecar** = mellékfolyamat / mellék-konténer ugyanahhoz a „szolgáltatáshoz” (K8s-ben gyakran ugyanabban a podban). Itt a **vLLM render**: `vllm launch render <HF-modell>` a hivatalos **vLLM CPU** image-ben (`vllm/vllm-openai-cpu`), nem a sim repo része.

- **Honnan a Qwen?** Hugging Face-ről (pl. `Qwen/Qwen2.5-1.5B-Instruct`): tokenizer + chat template (és ami a preprocesshez kell). Első induláskor letölti / cache-eli — ezért lehet lassú.
- **Mit csinál?** GPU nélküli preprocess: chat template alkalmazása + tokenizálás → token ID-k. **Nem** futtat inferenciát, **nem** generál választ (arra a sim mockja van).
- **Hol fut?** CPU-n (ezért `vllm-openai-cpu`). A sim `--render-url`-ön hívja (pl. `http://explore-render:8082`).
- Laptopon / Compose-ban külön konténer; K8s példában native sidecar.

**Egy render több simhez, vagy simenként egy render?**

- **Egy közös render + N sim:** kevesebb RAM/CPU; a simek ugyanarra a `--render-url`-re mutatnak, ugyanazzal a HF `--model`-lel. Tokenizálás / blokk-hash szempontjából mind „ugyanaz a modell”; a generálást továbbra is a simek színlelik. Router-méréshez általában **elég**.
- **Simenként saját render:** közelebb a Kubernetes példához (pod = sim + sidecar), de N-szeres terhelés. A routernél ritkán ad extra előnyt: az számít, **melyik sim milyen KV blokkokat tart**, nem az, hogy a tokenizálás 1 vagy N folyamatban fut.

Laptopon / Compose-ban először **1 render + N sim**-mel dolgozunk.

### `POD_IP` — miért kell?

A sim KV módban **ZMQ-n is közöl** cache-eseményeket (lásd lent). A topic névbe belekerül a pod azonosító, pl.:

```text
kv@<POD_IP>:<serving-port>@<model-név>
```

Példa: `kv@10.0.0.1:8000@Qwen/Qwen2.5-1.5B-Instruct`

Így egy külső router / KV index meg tudja mondani: *ez a blokk ezen a podon van*. K8s-ben `status.podIP`-ből jön (downward API). Notebookban pl. `POD_IP=127.0.0.1` elég. **Nélküle a sim KV módban el sem indul** — nem opcionális „szépítés”.

Megjegyzés: a notebookos `cached_tokens` méréshez **nem kell** ZMQ hallgató; a `POD_IP` akkor is kötelező, mert a sim enélkül nem bootol KV-vel.

### ZMQ PUB — hova és mit küld?

- **Mi:** ZeroMQ **PUB** socket — a sim *közzétesz* eseményeket; aki érdeklődik, **SUB**-ol.
- **Hova:** `--zmq-endpoint` (default `tcp://127.0.0.1:5557`). Docker/K8s-ben ez gyakran egy proxy / router felé mutat, nem feltétlenül a host localhostja.
- **Mit:** `BlockStored` (új blokkok a cache-ben), `BlockRemoved` (eviction), esetleg `AllBlocksCleared`. Payload: msgpack batch (hash-ek, token ID-k, …).
- **Kinek:** llm-d / KV-aware router rétegnek — hogy routing előtt tudja, melyik worker milyen prefixet tart. **Nem** része a `chat.completion` JSON-nak.

A kliens HTTP válasza és a ZMQ stream **párhuzamos csatornák**: az egyik a felhasználónak / benchmarknak, a másik a routereknek.

### Folyamat: request body → válasz

Röviden: **render tokenizál → sim cache-el / szimulál → HTTP válasz** (+ opcionális ZMQ).

```text
1. Kliens  →  POST /v1/chat/completions
              { model, messages, max_tokens, … }

2. Sim     →  átadja a szöveget a rendernek (--render-url)
              (chat template + tokenizer → token ID lista)

3. Sim     →  tokeneket --block-size méretű blokkokra vágja
              → megnézi a helyi prefix cache-t
              → cached_tokens = hány prompt-token már bent volt
              → új blokkokat eltárolja (ha kell, régieket kidob)

4. Sim     →  latency: TTFT / ITL (constant vagy per-token;
              cache hit a per-token TTFT-et csökkentheti)
              → mock szöveg (random / echo), usage mezőkkel

5. Sim     →  HTTP: chat.completion JSON
              (usage.prompt_tokens_details.cached_tokens, …)

6. Sim     →  (mellékág) ZMQ PUB: BlockStored / BlockRemoved
              topic: kv@POD_IP:port@model
              → router / index frissül (ha van hallgató)
```

A **kliens csak a sim HTTP portját** hívja; a renderhez közvetlenül nem beszél. A render a sim belső segédje tokenizáláshoz.

### Mit látunk a válaszban / metrikákban

- Válasz: `usage…cached_tokens`
- Metrikák: `vllm:prefix_cache_hits_total`, `prefix_cache_queries_total`, `kv_cache_usage_perc`
- ZMQ: `BlockStored` / `BlockRemoved` események külső routernek (nem a chat.completion része)

### `cache_hit_threshold`

Ha a helyi hit arány a küszöb alatt → üres válasz (`finish_reason: cache_threshold`). Ez **sim-oldali elutasítás** a kérés megérkezése után, nem előzetes router-súly.

### Összekeverhető fogalmak

| Dolog | Jelentés |
|-------|----------|
| `--enable-kvcache` | helyi prefix/KV, `cached_tokens`, ZMQ |
| transfer latency / `kv_transfer_params` | diszaggregált P/D átvitel |
| `enable-prefix-caching` | a sim **ignorálja** (mindig „enabled” szerűen viselkedik) |
| render sidecar | tokenizálás HF modellel; nem a válaszgenerátor |
| `POD_IP` + ZMQ topic | melyik pod tartja a blokkot — routernek |

---

## Egyéb funkciók (röviden)

Ezek nem kellenek az alap latency / router teszthez; a sim tudja őket, ha később kellenek. Részletek: [configuration.md](https://github.com/llm-d/llm-d-inference-sim/blob/main/docs/configuration.md).

- **Failure injection (szándékos hibák):** a sim időnként hibás választ adhat (pl. „túl sok kérés”, „szerverhiba”). Arra jó, hogy megnézzük: a kliens vagy a router hogyan bírja a hibákat.
- **Dataset válaszok (SQLite):** a sim nem csak random mondatokat adhat, hanem egy előre feltöltött beszélgetés-adatbázisból is válaszolhat (ha a kérdéshez van találat). Ehhez külön SQLite fájl kell — nem sima szöveges promptlista.
- **LoRA:** „ráépülő” kiegészítők a modellre; a sim memóriafoglalás nélkül utánozza, és jelzi a metrikákban — adapterek teszteléséhez.
- **Omni / kép:** ha be van kapcsolva, a szöveges válasz mellé néha (vagy kérésre) egy apró, hamis képet is tesz. Multimodális (szöveg+kép) kliensek kipróbálására.
- **SSL:** titkosított (HTTPS) kapcsolat a sim felé. Akkor kell, ha a környezet csak biztonságos csatornán beszél a szerverrel.
- **Fake metrics:** a sim nem a valódi belső számlálóit küldi a monitoringnak, hanem előre megadott, „színlelt” görbéket. Dashboardok és riasztások tesztelésére jó, amíg nincs valós forgalom. Alapból Prometheus `/metrics` az élő (szimulált) számlálókat adja.
- **Data parallel:** ugyanabból a simből több párhuzamos „másolat” (rank) egy gépen, különböző portokon. Nagyobb, több workeres elrendezés utánzására.
- **SGLang engine:** alapból a sim a **vLLM**-hez hasonlóan viselkedik; átkapcsolható egy másik motor (SGLang) utánzására. Kísérleti; sok funkció (pl. KV cache) ott nem elérhető.

---

## Notebook tapasztalatok ([explore_sim.ipynb](explore_sim.ipynb))

Amit a KV szakasz futtatásakor láttunk:

**Setup működik.** Egy render (Qwen) + egy sim (`--enable-kvcache`, `POD_IP`, `--render-url`) elég. A render első indulása lassú lehet (image/modell); a sim health gyorsabb. ZMQ listener **nem kellett** — a `cached_tokens` a válasz JSON-ból elég.

**Ugyanaz a prompt kétszer (egy sim).**  
1. kérés: `cached_tokens: 0` (üres cache).  
2. kérés: pl. `cached_tokens: 32` — a prompt blokkjai újrahasználva. A válasz *szövege* random módban ettől még más lehet; a cache a **prompt** újrafelhasználását jelzi, nem ugyanazt a választ.

**Két különböző user-szöveg (egy sim).**  
Akkor is lehet `cached_tokens > 0` (pl. `16`): a chat template / közös tokenek adnak legalább egy teljes blokkot (gyakori `block-size: 16`). Ez nem azt jelenti, hogy a két mondat megegyezik.

**Több sim.** A cache simenként külön van. Ugyanaz a prompt A-n majd B-n → B-n újra ~0. Azt, hogy melyik sim kapja a kérést, a **router** (vagy kézi URL) dönti el — a render nem routeol.

**Request mezők.**  
- `stream`: nem kötelező; `false` / kihagyva = egyben jön a JSON. `true` = darabokban (SSE); a notebook egylépéses JSON-olvasása ehhez nem elég.  
- `max_tokens`: kimeneti hossz felső korlátja; nem kötelező, de rövid teszthez hasznos.

**Összehasonlítás a régi `sample_chat_completion.json`-nel.** Ugyanaz a JSON-forma, de ott `fake-model` + `cached_tokens` mindig 0. A notebook KV futásánál HF modellnév, több `prompt_tokens` (template), és 2. kérésnél nem-nulla `cached_tokens`. A `kv_transfer_params: null` mindkét esetben rendben van (az P/D).

---

## Gyors emlékeztető

1. Sim = mock vLLM HTTP-n.  
2. YAML/CLI: model, mode, kontextus, concurrency, **latency → TTFT**.  
3. KV: `--enable-kvcache` + `POD_IP` + render + HF model.  
4. Flag-csere = újraindítás; request body = nem.  
5. Azonos prompt 2× → `cached_tokens` nő; más promptnál is lehet kis hit a közös template miatt.
