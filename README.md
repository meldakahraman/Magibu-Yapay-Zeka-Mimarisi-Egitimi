#  Magibu Uygulamalı Yapay Zeka Mimarisi Eğitimi - Ödev Havuzu

Bu depo, **Magibu** tarafından düzenlenen **Uygulamalı Yapay Zeka Mimarisi Eğitimi** boyunca her hafta hazırladığım ödevleri, projeleri ve uygulama çalışmalarını içermektedir.

---
##  Depo Dosya Dizini ve Mimari Yapısı

```plaintext
.1.Hafta Ödevler
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
├── 2.Hafta Ödevler/
│   ├── 1.Ödev Benchmark Testi/ 
│   │    └── Animasyon_Llama3_Benchmark_Testi.ipynb
│   │  
│   ├── 2.Ödev Kendi Benchmark Testini olusturma/
│   │   ├── animasyon_test_benchmark.json
│   │   ├── kendi_benchmark_testim.ipynb
│   │   ├── tum_modeller_benchmark_sonuclari.json
│   │   └── tum_modeller_optimize_benchmark_sonuclari.json
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
| **2.3. ödev** |[Animasyon Domain Llama-3 Fine-Tune]| `animasyon_lora_adapter `  | [HF Fine-Tune](https://huggingface.co/meldakahramann/animasyon-lora-adapter) |
| **3. ödev** | [Türkçe MMLU Benchmark Değerlendirme] |  `animasyon_lora_adapter ` | [HF Fine-Tune](https://huggingface.co/meldakahramann/animasyon-lora-adapter) |
| **4. ödev** | [Kendi Benchmark Testini Oluşturma] |  `kendi_benchmark_testim.ipynb ` | [HF Benchmark_Testi](https://huggingface.co/datasets/meldakahramann/animasyon-benchmark-dataset) |

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
---  


## 3. Ödev: Türkçe MMLU Benchmark Değerlendirme & Karşılaştırma Raporu

### Proje Hakkında
Fine-tune edilmiş modelimizin genel Türkçe anlama ve çoklu görev yeteneklerini test etmek amacıyla Türkçe MMLU (Massive Multitask Language Understanding) benchmark testi üzerinden kapsamlı bir değerlendirme gerçekleştirilmiştir. Modelimiz, türetildiği taban model ve farklı bir mimariye sahip açık kaynaklı model ile kıyaslanmıştır.

### Değerlendirme Metodolojisi
* **Benchmark Veri Seti:** `alibayram/yapay_zeka_turkce_mmlu_model_cevaplari` (**6.200 Soru**)
* **Değerlendirme Yöntemi:** Zero-shot prompting & Anlamsal Benzerlik (`paraphrase-multilingual-mpnet-base-v2`) destekli yanıt doğrulama
* **Donanım / Yapılandırma:** BitsAndBytes 4-bit NF4 Kuantizasyonu

### Türkçe MMLU Karşılaştırmalı Sonuçlar Tablosu

| Model Türü / Rolü | Model Reposu / Adı | Toplam Soru | Doğru Cevap | Başarı Oranı (Accuracy) | Değerlendirme Süresi | Soru Başına Ort. Süre |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Fine-Tuned Model (Bizim Modelimiz)** | `meldakahramann/animasyon-lora-adapter` | 6200 | 1261 | **%20.34** | 2797.9 sn (~46.6 dk) | ~0.45 sn |
| **Base Model** | `unsloth/llama-3-8b-bnb-4bit` | 6200 | 1244 | **%20.06** | 2700.9 sn (~45.0 dk) | ~0.43 sn |
| **Karşılaştırma Modeli** | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | 6200 | 1179 | **%19.01** | 2168.3 sn (~36.1 dk) | ~0.35 sn |

###  Detaylı Analiz ve Değerlendirmeler
1. **Fine-Tuning Başarısı (Doğruluk Oranı):** Fine-tune ettiğimiz modelimiz, 1261 doğru yanıt (%20.34 başarı oranı) elde ederek test edilen modeller arasında en yüksek doğruluğa ulaşmıştır. Türetildiği Base Llama-3-8B modeline kıyasla 17 soru daha fazla doğru yanıtlayarak başarısını %20.06'dan %20.34'e yükseltmiştir. Bu durum yapılan ince ayar (fine-tuning) sürecinin genel anlama kapasitesine olumlu katkı sağladığını kanıtlamaktadır.
2. **Çıkarım (Inference) Hızı ve Performans:** Fine-tuned modelimiz (2797.9 sn) ile Base model (2700.9 sn) benzer çıkarım süreleri sergilemiştir. Her iki model de soru başına ortalama ~0.43-0.45 saniye yanıt süresiyle stabil bir performans göstermiştir. Qwen 2.5 modeli daha hızlı çıkarma (2168.3 sn / ~36.1 dk) yapsa da doğruluk oranında (%19.01) geride kalmıştır.
3. **Genel Değerlendirme:** Uygulanan fine-tuning işlemi, modelin doğruluk performansını artırırken hız tarafında belirgin bir maliyet/yavaşlama yaratmamıştır. Llama-3 mimarisi üzerine inşa edilen modelimiz, geniş ölçekli Türkçe MMLU testinde hem base versiyonuna hem de Qwen 2.5 alternatifine karşı üstünlük kurmuştur.

---
## 4. Ödev: Kendi Benchmark Testini Oluşturma ve Model Kıyaslama

### Proje Hakkında
Animasyon alanında eğittiğimiz fine-tune modelimizin genelleme yeteneğini, bilgi birikimini ve diğer popüler açık kaynaklı büyük dil modelleriyle karşılaştırmak amacıyla 118 soruluk çoktan seçmeli animasyon benchmark testi (`animasyon_test_benchmark.json`) tasarlanmış ve detaylı model kıyaslama analizleri gerçekleştirilmiştir.

### Teknik Özellikler ve Mühendislik Yaklaşımları
* **Benchmark Veri Seti Mimarisi:** Sorular; animasyon filmlerinin konuları, seslendirme kadroları, stüdyoları, ödülleri ve tematik derinlikleri üzerinden zorluk derecelerine göre 4 şıklı (`A, B, C, D`) çeldiricilerle donatılarak 118 adet soru içerecek şekilde kurgulanmıştır.
* **Prompt Mühendisliği & Post-Processing:** LLM'lerin çoktan seçmeli sorularda açıklama yapma veya sohbet kalıntıları (`userassistant` vb.) üretme eğilimini engellemek adına prompt seviyesine sert kısıtlamalar eklenmiştir. Ayrıca Python `re` kütüphanesi kullanılarak metin içerisindeki bağımsız şık harflerini (`A, B, C, D`) hassasiyetle çeken bir Regex Filtreleme Katmanı sisteme entegre edilmiştir.
* **Çıkarım (Inference) Parametreleri:** Google Colab A100 GPU ortamında, 4-bit kuantalama (`bitsandbytes` NF4) ile 7B-9B parametre aralığındaki modeller test edilmiş; deterministik sonuçlar için `do_sample=False` ve `temperature=0.0` sabitlenmiştir.

###  Performans Karşılaştırma Matrisi (Animasyon Benchmark Sonuçları)

| Model Adı | Model Tipi / Repository | Ham Başarı Oranı (%) | Optimize Başarı Oranı (%) | Durum / Gözlem |
| :--- | :--- | :---: | :---: | :--- |
| **Gemma2_9B_Instruct** | `google/gemma-2-9b-it` | %73.73 | **%73.73** | Üstün ve kararlı bağlam yönetimi |
| **Kendi_FineTuned_Modelim** | `meldakahramann/animasyon-lora-adapter` | %39.83 | **%55.08** | Regex ve Prompt optimizasyonu ile ciddi artış |
| **Qwen2.5_7B_Instruct** | `Qwen/Qwen2.5-7B-Instruct` | %55.93 | **%55.08** | Dengeli ve kararlı instruct başarısı |
| **Mistral_7B_Instruct** | `mistralai/Mistral-7B-Instruct-v0.3` | %55.08 | **%52.54** | Standart sapma bandında seyir |
| **Llama3_8B_Base** | `unsloth/llama-3-8b-bnb-4bit` | %7.63 | **%26.27** | Teorik rastgele tahmin (%25) sınırına ulaşıldı |

---
## Eğitim Kazanımları

Bu eğitim süresince her hafta;
* Derin öğrenme ve modern yapay zeka mimarilerinin (Transformers, LLMs) arkasındaki teoriyi pratik kodlama ile birleştirmeyi,
* PyTorch kullanarak uçtan uca veri hatları, eğitim döngüleri ve model optimizasyonları yapmayı,
* Gerçek dünya problemlerine yapay zeka çözümleri üretmeyi öğreniyorum.

---
*Bu depo, Magibu Uygulamalı Yapay Zeka Mimarisi Eğitimi sürecince aktif olarak güncellenmektedir.*
