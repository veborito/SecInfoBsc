import base64
import os
import datetime
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.serialization import pkcs7, load_pem_private_key, load_pem_public_key
from cryptography import x509
from cryptography.x509.oid import NameOID

# differents chemins 
ALICE_PR_PATH = "./keys/alice_private_key.pem"
ALICE_PB_PATH = "./keys/alice_public_key.pem"

BOB_PR_PATH = "./keys/bob_private_key.pem"
BOB_PB_PATH = "./keys/bob_public_key.pem"

CERT_PATH = "./certs/exp_certificate.pem"

SIGN_PATH = "./signature.p7m"

ENC_MSG_PATH = "./enc_message.txt"
MSG_PATH = "./message.txt"
# information pour le certificat auto-signé

COUNTRY_NAME = "CH"
STATE_OR_PROVINCE_NAME = "Neuchâtel"
LOCALITY_NAME = "Neuchâtel"
ORGANIZATION_NAME = "unine"
COMMON_NAME = "www.unine.ch"


def create_certificate(private_key, public_key, cert_path):
    """Création d'un certificat auto-signé
	
	Keyword arguments:
	attributs -- tuple d'attributs contnenat les variables: 
 				 country_name, state_or_province_name, locality_name,
				 organizaton_name et common_name.
    private_key -- clé privée de l'expéditeur
    public_key -- clé publique de l'expéditeur
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
    """Fonction qui signe le hash du message avant de l'envoyer
    
    Keyword arguments:
    message -- message chiffré à signer
    cert -- certificat de l'expéditeur
    src_private -- clé privée de l'expéditeur

    """
    
    options =[pkcs7.PKCS7Options.DetachedSignature]
    with open(SIGN_PATH, "wb") as smime:
        smime.write(pkcs7.PKCS7SignatureBuilder().set_data(
            # le contenu chiffre du message au format octet
            base64.b64encode(message) # encode le message pour que openssl puisse le traiter correctement
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

def send(message, src_private_key, src_public_key, dst_public_key, msg_path):
    """Fonction qui "envoie" le message
    
    Keyword arguments:
    message -- message à chiffrer et signer
    src_private_key -- clé privée de l'expéditeur
    src_public_key -- clé publique de l'expéditeur
    dst_public_key -- clé publique du destinataire
    msg_path -- chemin vers le message
    
    Return: le message chiffré
    
    précision dans les étapes
    """
    cert = create_certificate(src_private_key, src_public_key, CERT_PATH)
    #chiffrement du message abec la clé publique du destinataire
    message_encrypted = dst_public_key.encrypt(message,
                                               padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), 
                                                            algorithm=hashes.SHA256(),label=None))
    create_signature(message_encrypted, cert, src_private_key)
    # on écrit le message chiffré dans un fichier .txt
    with open(msg_path, "wb") as msg:
        msg.write(base64.b64encode(message_encrypted)) # encode le message pour que openssl puisse le traiter correctement
    return message_encrypted

def verify(dst_private_key, encrypted_message):
    """Fonction verifiant la signature et déchiffrant le message chiffré
    
    Keyword arguments:
    argument -- description
    Return: return_description
    """
    # on vérifie la signature
    os.system("openssl smime -verify -in signature.p7m -inform SMIME -content enc_message.txt \
        -certfile certs/exp_certificate.pem -noverify")
    # déchiffrement avec la clé public du destinataire
    message = dst_private_key.decrypt(
        encrypted_message,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
            )
    )
    print("\n\nMessage: ", message.decode())

# Bob envoie un message a Alice en utilisant S/MIME.
def main():
    with open(MSG_PATH) as msg:
        message = msg.read().encode()
    with open(BOB_PR_PATH) as f:
        bob_private_key = load_pem_private_key(f.read().encode(), password=b"1234")
    with open(BOB_PB_PATH) as f:
        bob_public_key = load_pem_public_key(f.read().encode())
    with open(ALICE_PR_PATH) as f:
        alice_private_key = load_pem_private_key(f.read().encode(), password=b"1234")
    with open(ALICE_PB_PATH) as f:
        alice_public_key = load_pem_public_key(f.read().encode())
    # creation du certificat, chiffrement du message, signature du message chiffré
    encrypted_message = send(message, bob_private_key, bob_public_key, alice_public_key, ENC_MSG_PATH)
    # verification de la signature et certificat fourni. Déchiffrement du message
    verify(alice_private_key, encrypted_message)

if __name__ == "__main__":
	main()