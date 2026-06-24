#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <cstdint>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <filesystem>
#endif

namespace vinylizer {

#ifdef _WIN32

inline std::wstring utf8_to_wstring(const std::string& utf8) {
    if (utf8.empty()) return {};
    int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
    if (wlen <= 0) return {};
    std::wstring w(wlen, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, &w[0], wlen);
    w.resize(wlen - 1);
    return w;
}

inline std::vector<uint8_t> read_file_bytes(const std::string& utf8_path) {
    std::ifstream file(std::filesystem::path(utf8_to_wstring(utf8_path)),
                       std::ios::binary | std::ios::ate);
    if (!file) return {};
    auto size = file.tellg();
    file.seekg(0);
    std::vector<uint8_t> buf(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(buf.data()), size);
    return buf;
}

inline bool write_file_bytes(const std::string& utf8_path,
                              const void* data, size_t size) {
    std::ofstream file(std::filesystem::path(utf8_to_wstring(utf8_path)),
                       std::ios::binary);
    if (!file) return false;
    file.write(static_cast<const char*>(data), size);
    return file.good();
}

inline std::ifstream open_ifstream(const std::string& utf8_path,
                                     std::ios::openmode mode = std::ios::in) {
    return std::ifstream(std::filesystem::path(utf8_to_wstring(utf8_path)), mode);
}

inline std::ofstream open_ofstream(const std::string& utf8_path,
                                     std::ios::openmode mode = std::ios::out) {
    return std::ofstream(std::filesystem::path(utf8_to_wstring(utf8_path)), mode);
}

#else

inline std::vector<uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    auto size = file.tellg();
    file.seekg(0);
    std::vector<uint8_t> buf(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(buf.data()), size);
    return buf;
}

inline bool write_file_bytes(const std::string& path,
                              const void* data, size_t size) {
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;
    file.write(static_cast<const char*>(data), size);
    return file.good();
}

inline std::ifstream open_ifstream(const std::string& path,
                                     std::ios::openmode mode = std::ios::in) {
    return std::ifstream(path, mode);
}

inline std::ofstream open_ofstream(const std::string& path,
                                     std::ios::openmode mode = std::ios::out) {
    return std::ofstream(path, mode);
}

#endif

} // namespace vinylizer
