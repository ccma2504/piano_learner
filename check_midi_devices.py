import mido

print("--- MIDI 기기 진단 시작 ---")

try:
    backend = mido.backend
    print(f"사용 중인 Mido 백엔드: {backend.name}")
except Exception as e:
    print(f"Mido 백엔드를 가져오는 중 오류 발생: {e}")
    backend = None

if backend:
    try:
        input_devices = mido.get_input_names()
        
        if not input_devices:
            print("\n[결과] 연결된 MIDI 입력 기기를 찾을 수 없습니다.")
            print("다음 사항을 확인해 주세요:")
            print("1. 마스터 키보드가 컴퓨터에 제대로 연결되었는지 확인하세요.")
            print("2. 키보드의 전원이 켜져 있는지 확인하세요.")
            print("3. 다른 음악 프로그램(예: GarageBand)에서 기기가 인식되는지 확인해보세요.")
        else:
            print("\n[결과] 사용 가능한 MIDI 입력 기기 목록:")
            for i, device in enumerate(input_devices):
                print(f"{i}: {device}")

    except Exception as e:
        print(f"\nMIDI 기기 목록을 가져오는 중 오류 발생: {e}")
        print("Mido 또는 백엔드 라이브러리 설치에 문제가 있을 수 있습니다.")

print("\n--- MIDI 기기 진단 종료 ---")
