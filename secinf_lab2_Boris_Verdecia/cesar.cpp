#include <cctype>
#include <iostream>
#include <string>

static void encrypt(std::string &s, int i, int key, int lu) {
    if ((s[i] + key) > lu + 25)
      s[i] = lu + (s[i] + key - (lu + 26));
    else
      s[i] = s[i] + key;
}

static void decrypt(std::string &s, int i, int key, int lu) {
    if ((s[i] - key) < lu)
      s[i] = (lu + 26) - (lu - s[i]) - key; 
    else
      s[i] = s[i] - key;
}

static void cesar(std::string s, std::string &code, int key) {
  if (code.compare("encrypt") == 0) {
    bool is_alph;
    for (int i = 0; i < int(s.size()); i++) {
      is_alph = isalpha(s[i]); 
      if (is_alph and islower(s[i]))
        encrypt(s, i, key, 97);
      else if (is_alph)
        encrypt(s, i, key, 65);
    }
    std::cout << "The encrypted message is : " << s << "\n";
    return;
  }
  else if (code.compare("decrypt") == 0) {
    bool is_alph;
    for (int i = 0; i < int(s.size()); i++) {
      is_alph = isalpha(s[i]); 
      if (is_alph and islower(s[i]))
        decrypt(s, i, key, 97);
      else if (is_alph)
        decrypt(s, i, key, 65);
    }
    std::cout << "The decrypted message is : " << s << "\n";
    return;
  }
  std::cout << "Make sure to write encrypt or decrypt correctly !\n";
}

static void error_msg(){
  std::cout << "Error: Usage -> (message) optional : (encrypt/decrypt) (key)\n";
}

int main(int argc, char **argv) {
  if (argc == 2) {
    std::string string = argv[1];
    std::string code = "decrypt";
    for (int k = 1; k < 26; k++) {
      std::cout << "key = "<< k << ", ";
      cesar(string, code, k);
    }
    return 0;
  }
  else if (argc == 4) {
    std::string string = argv[1];
    std::string code = argv[2];
    int key = std::atoi(argv[3]);
    if (key > 25) {
      while (key > 25)
        key -= 26;
    }
    cesar(string, code, key);
    return 0;
  }
  error_msg();
  return 1;
}
