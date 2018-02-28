import time
import os
import logging
logger = logging.getLogger('root')


def setup_logger(output_dir, name):
    output_dir = expand_path(output_dir)
    log_filename = "{}.log".format(name)
    filepath = os.path.join(output_dir, log_filename)
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '[%(levelname)s] %(asctime)s:  %(message)s'
            },
            'simple': {
                'format': '[%(levelname)s]: %(message)s'
            }
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'simple',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'level': 'INFO',
                'class': 'logging.FileHandler',
                'formatter': 'standard',
                'filename': filepath,
                'mode': 'a',
            }
        },
        'loggers': {
            '': {
                'handlers': ['console', 'file'],
                'level': 'INFO',
                'propagate': True
            }
        }
    })

    logger.setLevel(logging.INFO)


def expand_path(path):
    """
    Expands a file path. This handles users and environmental variables.
    :param path: The path the expand
    :return: The expanded path
    """
    new_path = os.path.expanduser(path)
    return os.path.expandvars(new_path)


def clean_output(output_dir):
    """
    Creates the output directory if it does not exist and removes old files if it does.
    :param output_dir:
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))


def timed_invoke(action, method):
    """
    Invokes a method. Prints the start and finish with the invocation time.
    :param action: The string describing the method action
    :param method: The method to invoke
    :return: The return of the method
    """
    logger.info('Started {}...'.format(action))
    start = time.time()
    try:
        output = method()
        logger.info('Finished {} in {} seconds'.format(action, int(time.time() - start)))
        return output
    except Exception:
        logger.info('Exception while {} after {} seconds'.format(action, int(time.time() - start)))
        raise
