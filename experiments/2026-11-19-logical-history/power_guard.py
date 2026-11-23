"""Temporary AC-only idle-sleep request while an existing pilot process lives.
Does not change power plans, keep the display on, or override user-initiated sleep.
"""
import ctypes,sys,time,datetime
from ctypes import wintypes

class PowerStatus(ctypes.Structure):
    _fields_=[('ACLineStatus',wintypes.BYTE),('BatteryFlag',wintypes.BYTE),
        ('BatteryLifePercent',wintypes.BYTE),('SystemStatusFlag',wintypes.BYTE),
        ('BatteryLifeTime',wintypes.DWORD),('BatteryFullLifeTime',wintypes.DWORD)]

k=ctypes.WinDLL('kernel32',use_last_error=True)
k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
k.OpenProcess.restype=wintypes.HANDLE
k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
k.WaitForSingleObject.restype=wintypes.DWORD
k.CloseHandle.argtypes=[wintypes.HANDLE]
k.SetThreadExecutionState.argtypes=[wintypes.DWORD]
k.SetThreadExecutionState.restype=wintypes.DWORD
k.GetSystemPowerStatus.argtypes=[ctypes.POINTER(PowerStatus)]
handle=k.OpenProcess(0x00100000,False,int(sys.argv[1]))
if not handle:raise ctypes.WinError(ctypes.get_last_error())
previous=None
try:
    while k.WaitForSingleObject(handle,0)==258:
        status=PowerStatus()
        ac=bool(k.GetSystemPowerStatus(ctypes.byref(status)) and status.ACLineStatus==1)
        flags=0x80000000 | (0x00000001 if ac else 0)
        if not k.SetThreadExecutionState(flags):raise ctypes.WinError(ctypes.get_last_error())
        if ac!=previous:
            print(datetime.datetime.now().isoformat(),'AC idle-sleep request:',ac,flush=True)
            previous=ac
        time.sleep(30)
finally:
    k.SetThreadExecutionState(0x80000000)
    k.CloseHandle(handle)
    print(datetime.datetime.now().isoformat(),'Request released',flush=True)
