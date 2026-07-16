#  Magibu Uygulamalı Yapay Zeka Mimarisi Eğitimi - Ödev Havuzu

Bu depo, **Magibu** tarafından düzenlenen **Uygulamalı Yapay Zeka Mimarisi Eğitimi** boyunca her hafta hazırladığım ödevleri, projeleri ve uygulama çalışmalarını içermektedir.

---

##  Haftalık Ödevler ve Projeler

Aşağıdaki tablodan ilgili haftaya tıklayarak o haftanın konusuna, kodlarına ve detaylarına hızlıca ulaşabilirsiniz.

| Hafta | Ödev Konusu | Klasör / Dosya | Durum |
| :--- | :--- | :--- | :---: |
| **1. ödev** | [TinyGemma ile Türkçe İsim Türetme](#-1-hafta-tinygemma-ile-türkçe-isim-türetme-projesi) | `Tokenizer Oluşturma ve Minyatür Model Eğitimi` | 
| **2. ödev** | *Gelecek Ödev* | *Eklenecek* |  Beklemede |
| **3. ödev** | *Gelecek Ödev* | *Eklenecek* |  Beklemede |
| **4. ödev** | *Gelecek Ödev* | *Eklenecek* |  Beklemede |
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

---

## Eğitim Kazanımları

Bu eğitim süresince her hafta;
* Derin öğrenme ve modern yapay zeka mimarilerinin (Transformers, LLMs) arkasındaki teoriyi pratik kodlama ile birleştirmeyi,
* PyTorch kullanarak uçtan uca veri hatları, eğitim döngüleri ve model optimizasyonları yapmayı,
* Gerçek dünya problemlerine yapay zeka çözümleri üretmeyi öğreniyorum.

---
*Bu depo, Magibu Uygulamalı Yapay Zeka Mimarisi Eğitimi sürecince aktif olarak güncellenmektedir.*
