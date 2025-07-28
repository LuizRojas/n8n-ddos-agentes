from pydantic import BaseModel, Field
from typing import Dict, Any

class FeaturesInput(BaseModel):
    """
    Schema for input features for DDoS attack prediction.
    The field names and types must match the features used during model training
    after all preprocessing and feature engineering steps (from X.columns.tolist()).
    """
    # Features from your X.columns.tolist() output:
    Destination_Port: int = Field(..., description="Destination Port of the flow")
    Flow_Duration: int = Field(..., description="Duration of the flow in microseconds")
    Total_Fwd_Packets: int = Field(..., description="Total number of packets in the forward direction")
    Total_Backward_Packets: int = Field(..., description="Total number of packets in the backward direction")
    Total_Length_of_Fwd_Packets: float = Field(..., description="Total length of packets in the forward direction")
    Total_Length_of_Bwd_Packets: float = Field(..., description="Total length of packets in the backward direction")
    Fwd_Packet_Length_Max: float = Field(..., description="Maximum length of packets in the forward direction")
    Fwd_Packet_Length_Min: float = Field(..., description="Minimum length of packets in the forward direction")
    Fwd_Packet_Length_Mean: float = Field(..., description="Mean length of packets in the forward direction")
    Fwd_Packet_Length_Std: float = Field(..., description="Standard deviation of length of packets in the forward direction")
    Bwd_Packet_Length_Max: float = Field(..., description="Maximum length of packets in the backward direction")
    Bwd_Packet_Length_Min: float = Field(..., description="Minimum length of packets in the backward direction")
    Bwd_Packet_Length_Mean: float = Field(..., description="Mean length of packets in the backward direction")
    Bwd_Packet_Length_Std: float = Field(..., description="Standard deviation of length of packets in the backward direction")
    Flow_Bytes_s: float = Field(..., description="Number of bytes per second in the flow")
    Flow_Packets_s: float = Field(..., description="Number of packets per second in the flow")
    Flow_IAT_Mean: float = Field(..., description="Mean inter-arrival time of flows")
    Flow_IAT_Std: float = Field(..., description="Standard deviation of inter-arrival time of flows")
    Flow_IAT_Max: float = Field(..., description="Maximum inter-arrival time of flows")
    Flow_IAT_Min: float = Field(..., description="Minimum inter-arrival time of flows")
    Fwd_IAT_Total: float = Field(..., description="Total inter-arrival time of forward packets")
    Fwd_IAT_Mean: float = Field(..., description="Mean inter-arrival time of forward packets")
    Fwd_IAT_Std: float = Field(..., description="Standard deviation of inter-arrival time of forward packets")
    Fwd_IAT_Max: float = Field(..., description="Maximum inter-arrival time of forward packets")
    Fwd_IAT_Min: float = Field(..., description="Minimum inter-arrival time of forward packets")
    Bwd_IAT_Total: float = Field(..., description="Total inter-arrival time of backward packets")
    Bwd_IAT_Mean: float = Field(..., description="Mean inter-arrival time of backward packets")
    Bwd_IAT_Std: float = Field(..., description="Standard deviation of inter-arrival time of backward packets")
    Bwd_IAT_Max: float = Field(..., description="Maximum inter-arrival time of backward packets")
    Bwd_IAT_Min: float = Field(..., description="Minimum inter-arrival time of backward packets")
    Fwd_PSH_Flags: int = Field(..., description="Number of PSH flags in forward packets")
    Fwd_URG_Flags: int = Field(..., description="Number of URG flags in forward packets") # Note: Bwd PSH/URG flags not in this list
    Fwd_Header_Length: int = Field(..., description="Total bytes of IP/TCP headers in forward direction")
    Bwd_Header_Length: int = Field(..., description="Total bytes of IP/TCP headers in backward direction")
    Fwd_Packets_s: float = Field(..., description="Number of forward packets per second")
    Bwd_Packets_s: float = Field(..., description="Number of backward packets per second")
    Min_Packet_Length: float = Field(..., description="Minimum packet length in the flow")
    Max_Packet_Length: float = Field(..., description="Maximum packet length in the flow")
    Packet_Length_Mean: float = Field(..., description="Mean packet length in the flow")
    Packet_Length_Std: float = Field(..., description="Standard deviation of packet length in the flow")
    Packet_Length_Variance: float = Field(..., description="Variance of packet length in the flow")
    FIN_Flag_Count: int = Field(..., description="Number of FIN flags in the flow")
    SYN_Flag_Count: int = Field(..., description="Number of SYN flags in the flow")
    RST_Flag_Count: int = Field(..., description="Number of RST flags in the flow")
    PSH_Flag_Count: int = Field(..., description="Number of PSH flags in the flow")
    ACK_Flag_Count: int = Field(..., description="Number of ACK flags in the flow")
    URG_Flag_Count: int = Field(..., description="Number of URG flags in the flow")
    CWE_Flag_Count: int = Field(..., description="Number of CWE flags in the flow")
    ECE_Flag_Count: int = Field(..., description="Number of ECE flags in the flow")
    Down_Up_Ratio: float = Field(..., description="Ratio of download to upload packets")
    Average_Packet_Size: float = Field(..., description="Average size of packets in the flow")
    Avg_Fwd_Segment_Size: float = Field(..., description="Average size of forward segments")
    Avg_Bwd_Segment_Size: float = Field(..., description="Average size of backward segments")
    Fwd_Header_Length_1: int = Field(..., description="Forward Header Length (duplicate from dataset)") # Changed to _1
    Subflow_Fwd_Packets: int = Field(..., description="Number of packets in forward subflows")
    Subflow_Fwd_Bytes: float = Field(..., description="Number of bytes in forward subflows")
    Subflow_Bwd_Packets: int = Field(..., description="Number of packets in backward subflows")
    Subflow_Bwd_Bytes: float = Field(..., description="Number of bytes in backward subflows")
    Init_Win_bytes_forward: int = Field(..., description="Initial window size in forward direction")
    Init_Win_bytes_backward: int = Field(..., description="Initial window size in backward direction")
    act_data_pkt_fwd: int = Field(..., description="Count of packets with actual data in forward direction")
    min_seg_size_forward: int = Field(..., description="Minimum segment size in forward direction")
    Active_Mean: float = Field(..., description="Mean duration of active periods")
    Active_Std: float = Field(..., description="Standard deviation of active periods")
    Active_Max: float = Field(..., description="Maximum duration of active periods")
    Active_Min: float = Field(..., description="Minimum duration of active periods")
    Idle_Mean: float = Field(..., description="Mean duration of idle periods")
    Idle_Std: float = Field(..., description="Standard deviation of idle periods")
    Idle_Max: float = Field(..., description="Maximum duration of idle periods")
    Idle_Min: float = Field(..., description="Minimum duration of idle periods")
    Source_Port: int = Field(..., description="Source Port of the flow") # Added
    Protocol: int = Field(..., description="Protocol of the flow (e.g., 6 for TCP, 17 for UDP)") # Added
    client_real_ip: str = Field("UNKNOWN", description="Real client IP, injected for context.")


class PredictionOutput(BaseModel):
    """
    Schema for the output prediction from the DDoS detection API.
    """
    prediction: str = Field(..., description="Predicted class label (e.g., 'ATTACK' or 'BENIGN')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the prediction (0.0 to 1.0)")
    prediction_probabilities: Dict[str, float] = Field(..., description="Probabilities for each class label")
    is_attack: bool = Field(..., description="True if the predicted label indicates an attack, False otherwise")
    message: str = Field(..., description="A descriptive message about the prediction")