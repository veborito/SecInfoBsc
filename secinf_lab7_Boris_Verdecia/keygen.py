from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes, serialization

PUBLIC_EXPONENT = 65537
KEY_SIZE = 2048

# Très mauvaise pratique mais c'est pour l'exercice
BOB_PASSWORD = b"1234"
ALICE_PASSWORD = b"1234"

# differents chemins 
ALICE_PR_PATH = "./keys/alice_private_key.pem"
ALICE_PB_PATH = "./keys/alice_public_key.pem"

BOB_PR_PATH = "./keys/bob_private_key.pem"
BOB_PB_PATH = "./keys/bob_public_key.pem"

def keys_gen(public_exponent, key_size):
    """Fonction permettant la création d'un paire de clés (privée et publique)
	
	Keyword arguments:
	public_exponent -- Devrait être 65537 pour tout le monde!
 					   Utilsé pour s'assurer de la sécurité de l'algorithme de chiffrement.
	key_size -- Permet de choisir la taille de la clé privée en bits.
	Return: la paire de clés
	"""
	
    private_key = rsa.generate_private_key(
        public_exponent=public_exponent,
        key_size=key_size,
        backend=default_backend()
        )
    public_key = private_key.public_key()
    return private_key, public_key

def write_to_pem(private_key, public_key, password, pr_path, pb_path):
    """Fonction qui écrit la clé publique et la clé privé dans un fichier pem selon le chemin fourni
	
	Keyword arguments:
	private_key -- clé privée
	public_key -- clé publique
	password -- Mot de passe pour le chiffrement.
 				Attention le type de cette variable bytes
     			(exemple en python : password = b"password")
	pr_path -- chemin pour stocker la clé privée
 	pb_path -- chemin pour stocker la clé publique
	"""
	
    with open(pr_path, "wb") as f:
        f.write(private_key.private_bytes(
			encoding=serialization.Encoding.PEM,
			format=serialization.PrivateFormat.TraditionalOpenSSL,
			encryption_algorithm=serialization.BestAvailableEncryption(password),
		))
    with open(pb_path, "wb") as f:
        f.write(public_key.public_bytes(
			encoding=serialization.Encoding.PEM,
			format= serialization.PublicFormat.SubjectPublicKeyInfo,
		))

if __name__=="__main__":
    # Création de la clé privée et la clé publique pour bob et alice 
    bob_private_key, bob_public_key = keys_gen(PUBLIC_EXPONENT, KEY_SIZE)
    alice_private_key, alice_public_key =  keys_gen(PUBLIC_EXPONENT, KEY_SIZE)
	# On écrit les paires de clés dans des fichiers sur notre machine en format PEM
    write_to_pem(bob_private_key, bob_public_key, BOB_PASSWORD, BOB_PR_PATH, BOB_PB_PATH)
    write_to_pem(alice_private_key, alice_public_key, ALICE_PASSWORD, ALICE_PR_PATH, ALICE_PB_PATH)
