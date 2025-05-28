import { Github } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="bg-white shadow-inner mt-8">
      <div className="container mx-auto px-4 py-6">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <p className="text-gray-600 text-sm">
              &copy; {new Date().getFullYear()} FaceID. All rights reserved.
            </p>
          </div>
          <div className="flex space-x-4">
            <a
              href="#"
              className="text-gray-600 hover:text-primary-600 transition-colors duration-200"
              aria-label="GitHub"
            >
              <Github className="h-5 w-5" />
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;