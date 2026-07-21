#  Magibu Uygulamalı Yapay Zeka Mimarisi Eğitimi - Ödev Havuzu

Bu depo, **Magibu** tarafından düzenlenen **Uygulamalı Yapay Zeka Mimarisi Eğitimi** boyunca her hafta hazırladığım ödevleri, projeleri ve uygulama çalışmalarını içermektedir.

---
##  Depo Dosya Dizini ve Mimari Yapısı

```plaintext
.
├──  1.Ödev-Tokenizer Oluşturma ve Minyatür Model Eğitimi/
│   ├──  isimler.txt
│   └──  Tokenizer Oluşturma ve Minyatür Model Eğitimi.ipynb
│
├──  2.1.Ödev-Domain Veri Seti Oluşturma/
│   ├──  animasyon_dataset.json
│   ├──  veri_seti_olusturma.ipynb
│   └──  verimetni.txt
│
├──  2.2.Ödev- Özel BPE Tokenizer Oluşturma/
│   ├──  animasyon_bpe_tokenizer/
│   └──  tokenizer_olusturma.ipynb
│
├──  2.3.Ödev-Model Fine Tune Etme/
│   └──  animasyon_lora_adapter.ipynb
│
└──  README.md
```
---
##  Haftalık Ödevler ve Projeler

Aşağıdaki tablodan ilgili haftaya tıklayarak o haftanın konusuna, kodlarına ve detaylarına hızlıca ulaşabilirsiniz.

| Ödev NO | Ödev Konusu | Klasör / Dosya | Durum |
| :--- | :--- | :--- | :---: |
| **1.1. ödev** | [TinyGemma ile Türkçe İsim Türetme] | `Tokenizer Oluşturma ve Minyatür Model Eğitimi` | 
| **2.1. ödev** | [Animasyon Domain Veri Seti Hazırlama] | `veri_setini_olusturma.ipynb`, `animasyon_dataset.json` | [HF Dataset](https://huggingface.co/datasets/meldakahramann/animasyon-domain-dataset) | 
| **2.2. ödev** |[Özel BPE Tokenizer Oluşturma] | `tokenizer_olusturma.ipynb`, `animasyon_bpe_tokenizer/` | [HF Tokenizer](https://huggingface.co/meldakahramann/animasyon-bpe-tokenizer) |  |
| **2.3. ödev** |[Animasyon Domain Llama-3 Fine-Tune]| `animasyon_lora_adapter `  | [HF Dataset](https://huggingface.co/meldakahramann/animasyon-lora-adapter) |
| **5. ödev** | *Gelecek Ödev* | *Eklenecek* |  Beklemede |
---

##  1. Hafta: TinyGemma ile Türkçe İsim Türetme Projesi

### Proje Hakkında
Bu çalışmada, Türkçe isimlerin fonetik ve morfolojik yapılarını analiz etmek amacıyla sıfırdan bir **Byte Pair Encoding (BPE) Tokenizer** tasarlanmış ve minyatür Google **Gemma 4 (TinyGemma)** Transformer mimarisi kullanılarak yeni Türkçe isimler üretebilen bir dil modeli eğitilmiştir.

###  Teknik Özellikler ve Uygulama Adımları
* **BPE Tokenizer:** Türkçe isimler sonuna `</w>` eklenerek karakterlerine bölünmüş ve **35 birleştirme adımı** ile özel hece sözlüğü genişletilerek toplam **66 sözlük boyutuna ** ulaşılmıştır.
* **Veri Hattı:** İsimlerin başına `<unk>` (başlangıç belirteci) eklenmiş ve **`block_size = 6`** kayan pencere mekanizmasıyla PyTorch `Dataset` oluşturulmuştur. Paket boyutu **`batch_size = 5`** olarak optimize edilmiştir.
* **Model Mimarisi:** 128 hidden size, 4 katman (num_layers), 4 dikkat başlığı (num_heads) ve 2 KV başlığı (num_kv_heads) parametreleriyle TinyGemma kurulmuştur.
* **Eğitim:** AdamW (lr = 10^{-3}) optimizasyonu ile 100 epoch eğitilen model, **0.64** minimum kayıp (loss) seviyesine ulaşmıştır.
* **Üretim :** Olasılıksal örnekleme  ve **0.8 Sıcaklık** kullanılarak yaratıcı Türkçe isimler türetilmiştir.

### Örnek Üretim Sonuçları
* Girdi: `'b'` ➔ `belg`
* Girdi: `'e'` ➔ `esarp`
* Girdi: `'m'` ➔ `miraç`
* Girdi: `'s'` ➔ `sevil`
* Girdi: `'y'` ➔ `yağmur`

##  2.1. Ödev: Animasyon Domain Veri Seti Hazırlama

### Proje Hakkında
Büyük dil modellerinin spesifik alanlarda derinlemesine bilgi sahibi olabilmesi amacıyla sinema ve animasyon dünyasına (Pixar, DreamWorks, Studio Ghibli vb.) odaklanan özel bir **Instruction Dataset** tasarlanmıştır. Veri seti, modern yapay zeka akıl yürütme (reasoning) şablonlarına uygun biçimde yapılandırılmıştır.

### Teknik Özellikler
* **Veri Formatı:** Gelişmiş dil modellerinin standart chat şablonuna uygun olarak `messages` (`user` ve `assistant`) yapısında kurulmuştur.
* **Reasoning (İç Ses) Desteği:** `assistant` rolü içerisinde modelin yanıtı üretmeden önce gerçekleştirdiği mantıksal planlamayı simüle eden `thinking` alanları eklenmiştir.
* **Veri Havuzu İçeriği:** Popüler animasyon filmlerinin detaylı olay örgüleri, Türkçe seslendirme kadroları, yapım stüdyoları, ödül geçmişleri ve organik seyirci/eleştirmen yorumları harmanlanmıştır.
* **Çeşitlilik:** Her veri kategorisi için 3 farklı varyasyonda soru kalıpları ve kombinasyon soruları türetilerek toplam **1.020 satırlık** zengin bir veri kümesi oluşturulmuştur.

---

## 2.2. Ödev: Özel BPE Tokenizer Oluşturma

### Proje Hakkında
Hazırlanan animasyon veri setindeki Türkçe kelimelerin ve sinematik özel isimlerin (Örn: *Manny, Şrek, McQueen, Wall-E, Mononoke*) genel amaçlı tokenizer'lar tarafından gereksiz parçalanmasını önlemek amacıyla sıfırdan bir **Byte-Pair Encoding (BPE)** Tokenizer eğitilmiştir.

### Teknik Özellikler ve Uygulama Adımları
* **Eğitim Derlemi:** Model, Hugging Face Hub üzerinden doğrudan çekilen `animasyon_dataset.json` metin verileri (`content` ve `thinking` alanları dahil) kullanılarak eğitilmiştir.
* **Sözlük Yapılandırması:** Özel hece ve kelime yapısı analiz edilerek **128.000 hedef sözlük boyutu (vocab size)** tanımlanmış ve toplam **11.557 benzersiz token** başarıyla türetilmiştir.
* **Özel Belirteçler (Special Tokens):** Yapay zeka mimarisinin kararlı çalışması için `[UNK]`, `[PAD]`, `[CLS]`, `[SEP]`, `[MASK]` ve chat şablonu sınır belirteçleri olan `<|im_start|>`, `<|im_end|>` sisteme entegre edilmiştir.
* **Test Çıktısı:** Eğitilen model test edildiğinde sinematik kalıpları kusursuz biçimde sayısal ID dizilerine dönüştürebilmektedir:
  * **Metin:** `"Buz Devri filminin konusu nedir?"`
  * **Tokenlar:** `['Buz', 'Devri', 'filminin', 'konusu', 'nedir', '?']`
  * **Token ID'leri:** `[527, 633, 158, 534, 531, 28]`

---
## 2.3. Ödev: Animasyon Domain Llama-3 Fine-Tune
* **Amaç:** Hazırlanan animasyon veri seti ile Llama-3 8B Instruct taban modelini Unsloth ve LoRA (QLoRA) tekniklerini kullanarak ince ayardan (fine-tuning) geçirmek.
* **Teknik Özellikler**
   * **Taban Model:** unsloth/llama-3-8b-Instruct-bnb-4bit
   * **Eğitim Altyapısı:** Unsloth, SFTTrainer ve LoRA adaptörü (r=16, lora_alpha=16)
   * Eğitilen LoRA ağırlıkları meldakahramann/animasyon_lora_adapter reposuna aktarılmıştır.
### **Eğitim Kazanımları:**
* Modern Transformer mimarilerinin (Gemma, Llama-3) çalışma prensiplerini ve tokenizasyon süreçlerini sıfırdan uygulamayı,
* Belirli alanlara yönelik (Domain-Specific) sentetik/geliştirilmiş veri setleri hazırlamayı,
* Unsloth ve LoRA teknikleriyle düşük bellek tüketimiyle yüksek verimli LLM Fine-Tuning yapmayı tecrübe ettim.
* 

---

## Eğitim Kazanımları

Bu eğitim süresince her hafta;
* Derin öğrenme ve modern yapay zeka mimarilerinin (Transformers, LLMs) arkasındaki teoriyi pratik kodlama ile birleştirmeyi,
* PyTorch kullanarak uçtan uca veri hatları, eğitim döngüleri ve model optimizasyonları yapmayı,
* Gerçek dünya problemlerine yapay zeka çözümleri üretmeyi öğreniyorum.

---
*Bu depo, Magibu Uygulamalı Yapay Zeka Mimarisi Eğitimi sürecince aktif olarak güncellenmektedir.*
