class StatusTracker:
    
    '''
    Record the status of the previous few timestamps
    '''
    
    MAX_HISTORY = 10
    
    def __init__(self):
       
        self.history = []
        
    def add_status(self, status):
        
        """
        Add a new status to the history.
        
        Args:
            status (dict): A dictionary containing the status information.
        """
        if len(self.history) >= self.MAX_HISTORY:
            self.history.pop(0)
        self.history.append(status)
        