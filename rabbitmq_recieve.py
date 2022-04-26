import ssl
import pika, os

from content_based_algorithms.prefiller import PreFiller
from content_based_algorithms.tfidf import TfIdf

ssl_enabled = os.environ.get("ssl", False)

rabbitmq_user = os.environ.get('CLOUDAMQP_USERNAME', '***REMOVED***')
rabbitmq_password = os.environ.get('CLOUDAMQP_PASSWORD', '***REMOVED***')
rabbitmq_host = os.environ.get('CLOUDAMQP_HOST', '***REMOVED***')
rabbitmq_vhost = os.environ.get('CLOUDAMQP_VHOST', rabbitmq_user)


if ssl_enabled:
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(os.environ.get("ca_bundle", '/etc/pki/tls/certs/ca-bundle.crt'))
    ssl_options = pika.SSLOptions(context)
    port = os.environ.get("port", 5671)
else:
    ssl_options = None
    port = os.environ.get("port", 5672)

credentials = pika.credentials.PlainCredentials(
    username=rabbitmq_user, password=rabbitmq_password
)
connection_params = pika.ConnectionParameters(
    host=rabbitmq_host,
    ssl_options=ssl_options,
    port=port,
    credentials=credentials,
    virtual_host=rabbitmq_vhost,
    heartbeat=600,
    blocked_connection_timeout=300
)

rabbit_connection = pika.BlockingConnection(connection_params)

channel = rabbit_connection.channel()

channel.queue_declare(queue='new_articles_alert', durable=True)
print('[*] Waiting for messages. To exit press CTRL+C')


def callback(ch, method, properties, body):
    print("[x] Received %r" % body.decode())
    ch.basic_ack(delivery_tag=method.delivery_tag)
    if body.decode() == "new_articles_scrapped":
        print("Recieved message that new posts were scrapped.")
        print("I'm calling prefilling methods...")
        try:
            tf_idf = TfIdf()
            tf_idf.update_saved_matrix()
        except Exception as e:
            print("Exception occured" + str(e))


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='new_articles_alert', on_message_callback=callback)

channel.start_consuming()