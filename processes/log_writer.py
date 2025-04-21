from services.utils import initialize_logger

def log_writer_MP(stop_event, log_writer_queue):
    #setproctitle("PYlog_writer")
    logger1 = initialize_logger('facefinder/logs/systemLog_cam1')
    logger2 = initialize_logger('facefinder/logs/systemLog_cam2')
    
    loggers = {
            1: logger1,
            2: logger2
            }
    while not stop_event.is_set():
        log_type , log, CAM_ID = log_writer_queue.get(block=True)

        logger = loggers.get(CAM_ID)
        if logger:
            if log_type == "info":
                logger.info(log)
            elif log_type == "debug":
                logger.debug(log)
            elif log_type == "error":
                logger.error(log)