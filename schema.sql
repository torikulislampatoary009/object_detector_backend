-- The app creates these tables automatically on startup.
-- Run this file manually only if you prefer to set them up yourself.

CREATE TABLE IF NOT EXISTS detections (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    source        VARCHAR(20)  NOT NULL DEFAULT 'upload',
    filename      VARCHAR(255) NULL,
    image_width   INT          NOT NULL,
    image_height  INT          NOT NULL,
    pedestrians   INT          NOT NULL DEFAULT 0,
    cars          INT          NOT NULL DEFAULT 0,
    inference_ms  FLOAT        NOT NULL DEFAULT 0,
    created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS detection_boxes (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    detection_id  INT          NOT NULL,
    label         VARCHAR(30)  NOT NULL,
    confidence    FLOAT        NOT NULL,
    x1            FLOAT        NOT NULL,
    y1            FLOAT        NOT NULL,
    x2            FLOAT        NOT NULL,
    y2            FLOAT        NOT NULL,
    CONSTRAINT fk_detection FOREIGN KEY (detection_id)
        REFERENCES detections(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
