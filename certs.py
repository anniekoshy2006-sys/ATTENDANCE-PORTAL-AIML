"""
certs.py - helpers for LAN access.

Browsers only allow webcam access on HTTPS pages (or on http://localhost).
So that phones / other laptops on the same Wi-Fi can use the camera, the portal
serves HTTPS with a self-signed certificate generated here on first run.
"""
import datetime
import ipaddress
import json
import os
import socket

HERE = os.path.dirname(os.path.abspath(__file__))
CERT_DIR = os.path.join(HERE, "certs")
CERT_FILE = os.path.join(CERT_DIR, "portal.crt")
KEY_FILE = os.path.join(CERT_DIR, "portal.key")
META_FILE = os.path.join(CERT_DIR, "portal.json")


def lan_ips():
    """Best-effort list of this machine's LAN IPv4 addresses."""
    ips = set()
    try:  # the address used for the default route
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except Exception:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def ensure_cert(ips):
    """Create (or reuse) a self-signed cert valid for localhost + the given IPs."""
    wanted = sorted(set(ips) | {"127.0.0.1"})
    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE) and os.path.exists(META_FILE):
        try:
            with open(META_FILE) as f:
                if json.load(f).get("ips") == wanted:
                    return CERT_FILE, KEY_FILE
        except Exception:
            pass
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    os.makedirs(CERT_DIR, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Attendance Portal")])
    san = [x509.DNSName("localhost")] + [x509.IPAddress(ipaddress.ip_address(i)) for i in wanted]
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=825))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    with open(KEY_FILE, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM,
                                  serialization.PrivateFormat.TraditionalOpenSSL,
                                  serialization.NoEncryption()))
    with open(CERT_FILE, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(META_FILE, "w") as f:
        json.dump({"ips": wanted}, f)
    return CERT_FILE, KEY_FILE
