import { AlertCircle, X } from 'lucide-react';

interface BackendWarningProps {
  onClose: () => void;
}

const BackendWarning: React.FC<BackendWarningProps> = ({ onClose }) => {
  return (
    <div className="bg-warning-100 border-l-4 border-warning-500 p-4 fixed bottom-4 right-4 z-50 max-w-md rounded-md shadow-lg">
      <div className="flex">
        <div className="flex-shrink-0">
          <AlertCircle className="h-5 w-5 text-warning-500\" aria-hidden="true" />
        </div>
        <div className="ml-3 flex-1">
        </div>
        <div className="ml-auto pl-3">
          <div className="-mx-1.5 -my-1.5">
            <button
              onClick={onClose}
              className="inline-flex rounded-md p-1.5 text-warning-500 hover:bg-warning-200 focus:outline-none focus:ring-2 focus:ring-warning-600 focus:ring-offset-2"
            >
              <span className="sr-only">Dismiss</span>
              <X className="h-5 w-5" aria-hidden="true" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BackendWarning;