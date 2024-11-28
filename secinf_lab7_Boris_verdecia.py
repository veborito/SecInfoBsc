from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.serialization import pkcs7

PUBLIC_EXPONENT = 65537
KEY_SIZE = 2048

def keys_gen(public_exponent, key_size):
    """ Fonction permettant la création d'un paire de clés (privée et publique)
	
	arguments:
	public_exponent -- devrait être 65537 pour tout le monde!(Question de sécurité pour l'algorithme de chiffrement)
	key_size -- Permet de choisir la taille de la clé privée en bits.
	Return: la paire de clés
	"""
	
    private_key = rsa.generate_private_key(
        public_exponent=public_exponent,
        key_size=key_size,
        backend=default_backend()
        )
    public_key = private_key.public_key()
    return (private_key, public_key)

def write_to_pem(private_key, public_key, path_pr, path_pb):
    """ Fonction permettant l'écriture des clés
		dans un fichier pem.
	
	Keyword arguments:
	private_key -- clé privée
	public_key -- clé publique
	path -- chemin vers
 
	Return: return_description
	"""

    with open(path_pr, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.BestAvailableEncryption(b"passphrase"),
            ))



# Création de la clé privée et la clé publique de bob
bob_private_key, bob_public_key = keys_gen(PUBLIC_EXPONENT, KEY_SIZE)

# Création de la clé privée et la clé publique de alice
alice_pricvate_key, alice_public_key =  keys_gen(PUBLIC_EXPONENT, KEY_SIZE)

print(bob_private_key.private_bytes(

        encoding=serialization.Encoding.PEM,

        format=serialization.PrivateFormat.TraditionalOpenSSL,

        encryption_algorithm=serialization.BestAvailableEncryption(b"passphrase"),

    ))
