Odličan napredak sa skidanjem i pakovanjem sloja 58! 

Evo kompletne specifikacije za `.swan` bivektorske fajlove u dva oblika: **Kaitai Struct (`.ksy`)** za automatsko generisanje parsera u svim jezicima, i optimizovani **C++20 Zero-Copy Header (`.hpp`)** namenjen brzem parsiranju i mmap učitavanju u Ring-0 okruženju.

---

### 1. Kaitai Struct Specifikacija (`swan_file.ksy`)

Možete iskoristiti ovu specifikaciju za **Kaitai Struct Compiler (`ksc`)** radi generisanja parsera za C++, Python, Rust ili Deno/JS.

```yaml
meta:
  id: swan_file
  title: SWAN Bivector Container Format
  file-extension: swan
  endian: le
  license: MIT

doc: |
  SWAN binary format representation for bivector model weights.
  Supports optional JSON metadata catalog headers followed by packed Float32 bivector channels.

seq:
  - id: header_len
    type: u8
    doc: Length of JSON header catalog in bytes. If 0 or omitted, payload is raw Float32 data.

  - id: catalog_json
    type: str
    encoding: UTF-8
    size: header_len
    if: is_structured
    doc: Optional UTF-8 JSON metadata containing catalog parameters and tensor info.

  - id: bivectors
    type: f4
    repeat: eos
    doc: Continuous array of Float32 bivector coefficients (typically vocabulary_size * 35).

instances:
  is_structured:
    value: header_len > 0 and header_len < _io.size - 8
    doc: Evaluates whether the header contains a valid JSON catalog block.
```

---

### 2. C++20 Zero-Copy Structs & Mmap Loader (`swan_file.hpp`)

Ovaj C++ header podržava **zero-copy čitanje (`std::span` / `std::string_view`)** iz memorijski mapiranih fajlova (`mmap`), što omogućava trenutan pristup bivektorima bez alokacije dodatne memorije na heap-u:

```cpp
#ifndef SWAN_FILE_HPP
#define SWAN_FILE_HPP

#include <cstdint>
#include <vector>
#include <string>
#include <string_view>
#include <span>
#include <stdexcept>
#include <fstream>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace swan {

constexpr size_t TARGET_DIM = 35; // Standardna bivektorska dimenzija

// Binary Header (8 bajtova, Little-Endian)
#pragma pack(push, 1)
struct SwanHeader {
    uint64_t catalog_len; // Dužina JSON zaglavlja u bajtovima
};
#pragma pack(pop)

// Zero-Copy Memorijski Pogled na SWAN sloj
struct SwanView {
    uint64_t catalog_len = 0;
    std::string_view catalog_json;
    std::span<const float> bivectors;

    // Vraća broj tokena u bivektorskoj matrici
    [[nodiscard]] size_t vocab_size(size_t dim = TARGET_DIM) const noexcept {
        return bivectors.size() / dim;
    }

    // Izvlači 35-bitni bivektorski slajs za zadati token ID bez kopiranja
    [[nodiscard]] std::span<const float> get_bivector(size_t token_id, size_t dim = TARGET_DIM) const noexcept {
        size_t offset = token_id * dim;
        if (offset + dim <= bivectors.size()) {
            return bivectors.subspan(offset, dim);
        }
        return {};
    }
};

// Funkcija za parsiranje sirovog memorijskog bafera (RAM / mmap)
inline SwanView parse_swan_buffer(const uint8_t* data, size_t total_bytes) {
    if (!data || total_bytes < sizeof(SwanHeader)) {
        throw std::runtime_error("[SWAN C++] Nevažeća veličina bafera.");
    }

    SwanView view;
    const auto* header = reinterpret_cast<const SwanHeader*>(data);
    
    // Provera da li postoji JSON zaglavlje
    if (header->catalog_len > 0 && (header->catalog_len + sizeof(SwanHeader)) <= total_bytes) {
        view.catalog_len = header->catalog_len;
        view.catalog_json = std::string_view(
            reinterpret_cast<const char*>(data + sizeof(SwanHeader)),
            header->catalog_len
        );

        const size_t data_offset = sizeof(SwanHeader) + header->catalog_len;
        const size_t float_bytes = total_bytes - data_offset;
        view.bivectors = std::span<const float>(
            reinterpret_cast<const float*>(data + data_offset),
            float_bytes / sizeof(float)
        );
    } else {
        // Sirovi Float32 stream bez JSON zaglavlja
        view.catalog_len = 0;
        view.bivectors = std::span<const float>(
            reinterpret_cast<const float*>(data),
            total_bytes / sizeof(float)
        );
    }

    return view;
}

#if defined(__linux__) || defined(__APPLE__)
// POSIX mmap za trenutan zero-copy pristup sa diska
class MmappedSwanFile {
private:
    int fd_ = -1;
    size_t length_ = 0;
    void* addr_ = MAP_FAILED;
    SwanView view_;

public:
    explicit MmappedSwanFile(const std::string& filepath) {
        fd_ = open(filepath.c_str(), O_RDONLY);
        if (fd_ == -1) {
            throw std::runtime_error("[SWAN C++] Ne mogu otvoriti fajl: " + filepath);
        }

        struct stat sb;
        if (fstat(fd_, &sb) == -1) {
            close(fd_);
            throw std::runtime_error("[SWAN C++] Greška pri čitanju veličine fajla.");
        }
        length_ = sb.st_size;

        addr_ = mmap(nullptr, length_, PROT_READ, MAP_SHARED, fd_, 0);
        if (addr_ == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("[SWAN C++] Greška pri mmap alokaciji.");
        }

        view_ = parse_swan_buffer(static_cast<const uint8_t*>(addr_), length_);
    }

    ~MmappedSwanFile() {
        if (addr_ != MAP_FAILED) munmap(addr_, length_);
        if (fd_ != -1) close(fd_);
    }

    // Onemogućeno kopiranje radi bezbednosti pokazivača
    MmappedSwanFile(const MmappedSwanFile&) = delete;
    MmappedSwanFile& operator=(const MmappedSwanFile&) = delete;

    [[nodiscard]] const SwanView& view() const noexcept { return view_; }
};
#endif

} // namespace swan

#endif // SWAN_FILE_HPP
```

---

### Primer upotrebe u C++20:

```cpp
#include "swan_file.hpp"
#include <iostream>

int main() {
    try {
        // Instant zero-copy čitanje layer_58.swan preko mmap-a
        swan::MmappedSwanFile swan_layer("layer_58.swan");
        const auto& view = swan_layer.view();

        std::cout << "Učitan SWAN sloj!" << std::endl;
        std::cout << "Ukupno tokena: " << view.vocab_size() << std::endl;

        // Dobavljanje 35-bitnog bivektora za Token #100 u nultom vremenu
        auto vec100 = view.get_bivector(100);
        if (!vec100.empty()) {
            std::cout << "Bivektor #100: " << vec100 << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Greška: " << e.what() << std::endl;
    }
    return 0;
}
```

⚡ Mogu napisati i brzu C++ test skriptu (`make_test.sh`) koja kompajlira ovaj modul sa `g++ -std=c++20` za proveru brzine čitanja sloja 58!