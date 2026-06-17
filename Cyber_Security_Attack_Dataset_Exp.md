# Cyber Security Attack Dataset Explanation

## Dataset Overview

### English

This dataset contains 100,000 synthetic network traffic records simulating common network behaviors and cyber attack patterns, including Distributed Denial of Service (DDoS) attacks, brute-force login attempts, and port scanning activities. Each row represents a single connection session, containing metrics such as session duration, bytes transferred, packet counts, protocol types, failed login attempts, and the attack type classification label.

The dataset is suitable for exploratory data analysis, traffic pattern identification, feature correlation studies, and training machine learning classification models for intrusion detection systems (IDS).

### Bahasa Indonesia

Dataset ini berisi 100.000 data lalu lintas jaringan sintetis yang mensimulasikan perilaku jaringan umum dan pola serangan siber, termasuk serangan Distributed Denial of Service (DDoS), upaya login brute-force, dan aktivitas pemindaian port (port scanning). Setiap baris mewakili satu sesi koneksi, yang mencakup metrik seperti durasi sesi, jumlah byte yang ditransfer, jumlah paket, tipe protokol, jumlah kegagalan login, dan label klasifikasi jenis serangan.

Dataset ini cocok digunakan untuk analisis data eksploratif (EDA), identifikasi pola lalu lintas jaringan, studi korelasi fitur, serta pelatihan model machine learning klasifikasi untuk sistem deteksi intrusi (IDS).

## Dataset Structure

- Rows: 100,000
- Columns: 7
- Granularity: connection-level network traffic logs
- Main categorical features: `protocol`, `attack_type`
- Target label for classification: `attack_type`
- Key traffic volume metrics: `src_bytes`, `dst_bytes`, `packet_count`

## Column Explanation

| Column | Data Type | English Explanation | Penjelasan Bahasa Indonesia | Analysis Notes |
| --- | --- | --- | --- | --- |
| `duration` | Integer | Duration of the connection in seconds. | Durasi koneksi dalam satuan detik. | DDoS and PortScan connections are typically very short, while Normal and BruteForce are longer. |
| `src_bytes` | Integer | Number of bytes sent from source to destination. | Jumlah byte yang dikirim dari sumber ke tujuan. | Crucial metric for identifying high-volume DDoS attacks. |
| `dst_bytes` | Integer | Number of bytes sent from destination to source. | Jumlah byte yang dikirim dari tujuan ke sumber. | Useful for understanding bidirectional payload distributions. |
| `packet_count` | Integer | Total number of packets transmitted during the session. | Total jumlah paket data yang dikirim selama sesi. | DDoS attacks exhibit extremely high packet counts compared to other traffic types. |
| `protocol` | String | Network protocol used (TCP or UDP). | Protokol jaringan yang digunakan (TCP atau UDP). | DDoS attacks are exclusively on TCP, while BruteForce attacks are exclusively on UDP in this dataset. |
| `failed_logins` | Integer | Number of failed login attempts during the connection. | Jumlah upaya login yang gagal selama koneksi. | Directly identifies BruteForce attacks, as all other classes have exactly zero failed logins. |
| `attack_type` | String | Classification label of the traffic (Normal, DDoS, PortScan, BruteForce). | Label klasifikasi dari lalu lintas jaringan (Normal, DDoS, PortScan, BruteForce). | The target variable for intrusion detection modeling. |

## Suggested Analysis Questions

### English

1. What is the total count of connections for each type of traffic (Normal vs. attacks)?
2. How do the TCP and UDP protocols compare in terms of safety and security?
3. Which attack type has the highest network throughput (bytes per second)?
4. How does the average packet size (bytes per packet) differ across various attack types?
5. Is there a strong linear correlation between the volume metrics (`src_bytes`, `dst_bytes`, `packet_count`)?
6. Can we uniquely identify BruteForce attacks using the `failed_logins` feature?
7. How do DDoS attacks differ from Normal traffic in terms of duration and bytes transferred?
8. Does PortScan traffic mimic Normal traffic in terms of throughput?

### Bahasa Indonesia

1. Berapa jumlah total koneksi untuk setiap jenis lalu lintas jaringan (Normal vs. serangan)?
2. Bagaimana perbandingan kerentanan keamanan antara protokol TCP dan UDP?
3. Jenis serangan mana yang memiliki throughput (byte per detik) tertinggi?
4. Bagaimana perbedaan ukuran rata-rata paket data (byte per paket) di berbagai jenis serangan?
5. Apakah terdapat korelasi linier yang kuat antara metrik volume jaringan (`src_bytes`, `dst_bytes`, `packet_count`)?
6. Apakah kita dapat mengidentifikasi serangan BruteForce secara unik hanya menggunakan fitur `failed_logins`?
7. Bagaimana perbedaan serangan DDoS dengan lalu lintas Normal dalam hal durasi dan byte yang ditransfer?
8. Apakah lalu lintas PortScan meniru pola lalu lintas Normal dalam hal throughput jaringan?

## Useful Derived Features

### English

You can create additional features from this dataset:

| Derived Feature | Formula / Idea | Purpose |
| --- | --- | --- |
| `bytes_per_packet` | `src_bytes / packet_count` | Measures average packet payload size. Useful for identifying anomalous data volume configurations. |
| `total_bytes` | `src_bytes + dst_bytes` | Measures total bidirectional data volume of the connection. |
| `troughput` | `(src_bytes + dst_bytes) / duration` | Measures transmission speed in bytes per second. Highly effective for identifying DDoS flood attacks. |
| `is_attack` | `attack_type != 'Normal'` | Simplifies the target column into binary classification (Normal vs. Attack). |

### Bahasa Indonesia

Kamu bisa membuat fitur tambahan dari dataset ini:

| Fitur Turunan | Formula / Ide | Tujuan |
| --- | --- | --- |
| `bytes_per_packet` | `src_bytes / packet_count` | Mengukur ukuran rata-rata paket data. Berguna untuk mengidentifikasi pola volume payload yang janggal. |
| `total_bytes` | `src_bytes + dst_bytes` | Mengukur total volume data dua arah dalam koneksi jaringan. |
| `troughput` | `(src_bytes + dst_bytes) / duration` | Mengukur kecepatan transfer data dalam byte per detik. Sangat efektif untuk mengidentifikasi serangan banjir DDoS. |
| `is_attack` | `attack_type != 'Normal'` | Menyederhanakan label target menjadi klasifikasi biner (Normal vs. Serangan). |

## Important Notes

### English

- This dataset is connection-level, meaning each row represents a single network session log, not a customer transaction or system log.
- The dataset is synthetic (simulated data) as indicated by highly balanced distributions, so insights represent exploratory pattern analysis rather than actual real-world production networks.
- `failed_logins` is a very strong feature for BruteForce, but in real networks, failed logins can also happen in normal scenarios (user typos), so the perfect separation in this dataset might be due to its synthetic nature.
- Anomaly detection based on throughput is highly effective for DDoS, but signature-based analysis (like tracking failed login counts) is required for BruteForce and PortScan.

### Bahasa Indonesia

- Dataset ini berada pada level koneksi, sehingga setiap baris mewakili satu log sesi jaringan, bukan transaksi pelanggan atau log sistem lokal.
- Dataset ini bersifat sintetis (simulasi data) yang terlihat dari distribusi kategori yang sangat seimbang, sehingga temuan di dalamnya mewakili analisis pola eksploratif daripada jaringan produksi dunia nyata.
- `failed_logins` adalah fitur yang sangat kuat untuk mendeteksi BruteForce, namun di jaringan nyata, kegagalan login juga dapat terjadi pada skenario normal (pengguna salah ketik), sehingga pemisahan sempurna di dataset ini terjadi karena sifat sintetisnya.
- Deteksi anomali berbasis throughput sangat efektif untuk serangan DDoS, tetapi analisis berbasis tanda (seperti melacak jumlah gagal login) diperlukan untuk mendeteksi BruteForce dan PortScan.
