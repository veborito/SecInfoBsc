import os
import getpass
import datetime
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.serialization import pkcs7
from cryptography import x509
from cryptography.x509.oid import NameOID

PUBLIC_EXPONENT = 65537
KEY_SIZE = 2048

# Pour ne pas avoir de mot de passe hardcodé 
BOB_PASSWORD = getpass.getpass("Password for Bob:").encode()
ALICE_PASSWORD = getpass.getpass("Password for Alice:").encode()

# differents chemins 
ALICE_PR_PATH = "./keys/alice_private_key.pem"
ALICE_PB_PATH = "./keys/alice_public_key.pem"

BOB_PR_PATH = "./keys/bob_private_key.pem"
BOB_PB_PATH = "./keys/bob_public_key.pem"

CERT_PATH = "./certs/exp_certificate.pem"

SIGN_PATH = "./signature.p7m"
# information pour le certificat auto-signé

COUNTRY_NAME = "CH"
STATE_OR_PROVINCE_NAME = "Neuchâtel"
LOCALITY_NAME = "Neuchâtel"
ORGANIZATION_NAME = "unine"
COMMON_NAME = "www.unine.ch"

def keys_gen(public_exponent, key_size):
    """Fonction permettant la création d'un paire de clés (privée et publique)
	
	arguments:
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
    """ Fonction qui écrit la clé publique et la clé privé dans un fichier pem selon le chemin fourni
	
	arguments:
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

def create_certificate(private_key, public_key, cert_path):
    """ Création d'un certificat auto-signé
	
	arguments:
	attributs -- tuple d'attributs contnenat les variables: 
 				 country_name, state_or_province_name, locality_name,
				 organizaton_name et common_name.
    private_key
    public_key
    cert_path -- path pour le certificat en format .pem
	"""
 
    subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COUNTRY_NAME, COUNTRY_NAME),
    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, STATE_OR_PROVINCE_NAME),
    x509.NameAttribute(NameOID.LOCALITY_NAME, LOCALITY_NAME),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, ORGANIZATION_NAME),
    x509.NameAttribute(NameOID.COMMON_NAME, COMMON_NAME),
    ])
    cert = x509.CertificateBuilder().subject_name(
		subject
	).issuer_name(
		issuer
	).public_key(
		public_key
	).serial_number(
		x509.random_serial_number()
	).not_valid_before(
		datetime.datetime.now(datetime.timezone.utc)
	).not_valid_after(
		# Our certificate will be valid for 10 days
		datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=10)
	).add_extension(
		x509.SubjectAlternativeName([x509.DNSName("localhost")]),
		critical=False,
	# Sign our certificate with our private key
	).sign(private_key, hashes.SHA256())
	# Write our certificate out to disk.
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    return cert

def create_signature(message, cert, src_private_key):
    options =[pkcs7.PKCS7Options.DetachedSignature]
    with open(SIGN_PATH, "wb") as smime:
        smime.write(pkcs7.PKCS7SignatureBuilder().set_data(
            # le contenu chiffre du message au format octet
            message
            # fournir des informations sur le signataire
        ).add_signer(
            certificate=cert,
            private_key=src_private_key,
            hash_algorithm=hashes.SHA256()
        # signer le message
        ).sign(
            encoding=serialization.Encoding.SMIME,
            options=options
        ))

def send(message, src_private_key, src_public_key, dst_public_key):
    cert = create_certificate(src_private_key, src_public_key, CERT_PATH)
    message_encrypted = dst_public_key.encrypt(message,
                                               padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), 
                                                            algorithm=hashes.SHA256(),label=None))
    create_signature(message_encrypted, cert, src_private_key)
    return message_encrypted

#def verify(message_encrypted, dst_private_key, dst_public_key, src_private_key, cert_path, sign_path):


def main():
    message = b"Hello World!"
    # Création de la clé privée et la clé publique pour bob et alice 
    bob_private_key, bob_public_key = keys_gen(PUBLIC_EXPONENT, KEY_SIZE)
    alice_private_key, alice_public_key =  keys_gen(PUBLIC_EXPONENT, KEY_SIZE)
	# On écrit les paires de clés dans des fichiers sur notre machine en format PEM
    write_to_pem(bob_private_key, bob_public_key, BOB_PASSWORD, BOB_PR_PATH, BOB_PB_PATH)
    write_to_pem(alice_private_key, alice_public_key, ALICE_PASSWORD, ALICE_PR_PATH, ALICE_PB_PATH)
    
    message_encrypted = send(message, bob_private_key, bob_public_key, alice_public_key)
    print(message_encrypted)
	
if __name__ == "__main__":
	main()